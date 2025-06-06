import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from models import EmbeddingModel, LanguageModel
from build_index import VectorIndexBuilder
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self,
                 index_dir: str = "./index",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 ollama_model: str = "gemma2:2b",
                 hf_model: str = "google/gemma-2-2b",
                 llm_api_token: Optional[str] = None):
        """
        Query engine for RAG pipeline. Loads FAISS index, embeds queries, retrieves context, and calls LLM.
        """
        self.index_dir = index_dir
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
        self.llm = LanguageModel(ollama_model=ollama_model, hf_model=hf_model, api_token=llm_api_token)
        self.index = VectorIndexBuilder(model_name=embedding_model_name, index_dir=index_dir)
        logger.info(f"QueryEngine initialized with index at {index_dir}")

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Embed the user query, retrieve top-k chunks, construct prompt, and get LLM answer.
        Returns answer, context, and metadata.
        """
        logger.info(f"Received query: {user_query}")
        # 1. Embed query
        query_embedding = self.embedding_model.embed([user_query])[0]
        query_embedding = np.array(query_embedding) / (np.linalg.norm(query_embedding) + 1e-8)
        # 2. Search index
        results = self.index.search(user_query, top_k=top_k)
        if not results:
            logger.warning("No relevant chunks found in index.")
            return {"answer": "No relevant information found.", "sources": [], "context": ""}
        # 3. Build context - Retrieve actual chunk text here
        context_chunks = [self.index.get_chunk_text(r.get('chunk_id')) for r in results if r.get('chunk_id')]
        # Filter out None values if any chunk text was not found
        context_chunks = [chunk for chunk in context_chunks if chunk]
        context = "\n---\n".join(context_chunks)
        # 4. Build prompt
        prompt = self._build_prompt(context, user_query)
        # 5. Call LLM
        answer = self.llm.generate(prompt)
        # 6. Return answer and sources
        # Prepare enriched sources for display
        enriched_sources = []
        for result_item in results:
            chunk_id = result_item.get('chunk_id')
            chunk_text = 'Content not available' # Default value
            document_title = result_item.get('document_title', 'Unknown Document') # Default value
            page_number = 'Unknown Page' # Default value
            
            if chunk_id:
                # Attempt to get chunk text
                retrieved_text = self.index.get_chunk_text(chunk_id)
                if retrieved_text is not None:
                    chunk_text = retrieved_text
                
                # Attempt to get metadata
                metadata = result_item.get('chunk_metadata', {})
                # Check for document title/path in metadata as fallback
                if document_title == 'Unknown Document':
                     document_title = metadata.get('document_title', metadata.get('document_path', 'Unknown Document'))

                # Get page number from metadata
                page_number = metadata.get('page_number', 'Unknown Page')
                
            enriched_sources.append({
                'chunk_text': chunk_text,
                'source': document_title,
                'page': page_number,
                'score': result_item.get('similarity_score')
            })
                
        return {
            "answer": answer,
            "sources": enriched_sources, # Return the enriched sources
            "context": context # Keep context for now, might be useful later or for debugging
        }

    def _build_prompt(self, context: str, user_query: str) -> str:
        prompt = (
            "You are an expert assistant. Use the following context from scientific papers to answer the user's question.\n"
            f"Context:\n{context}\n"
            f"Question: {user_query}\n"
            "Answer:"
        )
        return prompt

    def arxiv_lookup(self, description: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Look up a paper on arXiv based on user description.
        Returns a list of top results (title, authors, abstract, link).
        """
        logger.info(f"Arxiv lookup for: {description}")
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{description}",
            "start": 0,
            "max_results": max_results
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip()
                link = entry.find('atom:id', ns).text.strip()
                authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]
                results.append({
                    'title': title,
                    'authors': authors,
                    'summary': summary,
                    'link': link
                })
            return {"results": results}
        except Exception as e:
            logger.error(f"Arxiv API error: {e}")
            return {"error": str(e), "results": []}

# Example usage (for testing, not for production UI)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="User query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_dir", type=str, default="./index")
    args = parser.parse_args()
    engine = QueryEngine(index_dir=args.index_dir)
    result = engine.query(args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2)) 