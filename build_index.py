"""
Vector Index Builder for Document Q&A AI Agent
Embeds document chunks and builds FAISS index for efficient retrieval
Supports incremental updates and optimized search performance
"""

import faiss
import numpy as np
import pickle
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from pdf_parser import parse_pdf
from utils import intelligent_chunking, generate_file_hash, setup_logging


@dataclass
class IndexMetadata:
    """Metadata for the vector index"""
    index_version: str = "1.0"
    created_at: str = ""
    last_updated: str = ""
    total_documents: int = 0
    total_chunks: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    document_hashes: Dict[str, str] = None

    def __post_init__(self):
        if self.document_hashes is None:
            self.document_hashes = {}


@dataclass
class ChunkMetadata:
    """Metadata for individual text chunks"""
    chunk_id: str
    document_id: str
    document_title: str
    section_title: Optional[str]
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    chunk_type: str  # 'text', 'table', 'equation', 'abstract'
    page_number: Optional[int] = None
    section_type: Optional[str] = None
    # Note: We won't store the text directly in metadata to keep this file small

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VectorIndexBuilder:
    """
    Builds and manages FAISS vector index for document retrieval
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 index_dir: str = "./index",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the vector index builder

        Args:
            model_name (str): SentenceTransformer model name
            index_dir (str): Directory to save index files
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.logger = setup_logging()
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create index directory
        self.index_dir.mkdir(exist_ok=True)

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = None
        self.chunk_metadata: List[ChunkMetadata] = []
        self.chunk_texts: Dict[str, str] = {}
        self.index_metadata = IndexMetadata(
            embedding_model=model_name,
            embedding_dimension=self.embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=datetime.now().isoformat()
        )

        # Load existing index if available
        self._load_existing_index()

    def _load_existing_index(self):
        """Load existing index, metadata, and chunk texts if available"""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.json"
        chunks_metadata_path = self.index_dir / "chunks_metadata.json"
        chunks_text_path = self.index_dir / "chunks_text.json"

        if all(path.exists() for path in [index_path, metadata_path, chunks_metadata_path, chunks_text_path]):
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    self.index_metadata = IndexMetadata(**metadata_dict)

                # Load chunk metadata
                with open(chunks_metadata_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    self.chunk_metadata = [ChunkMetadata(**chunk) for chunk in chunks_data]

                # Load chunk texts
                with open(chunks_text_path, 'r', encoding='utf-8') as f:
                    self.chunk_texts = json.load(f)

                self.logger.info(f"Loaded existing index with {len(self.chunk_metadata)} chunks and {len(self.chunk_texts)} chunk texts")

            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}")
                self._initialize_new_index()
        else:
            self._initialize_new_index()

    def _initialize_new_index(self):
        """Initialize a new FAISS index and clear chunk texts"""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.chunk_metadata = []
        self.chunk_texts = {}
        self.index_metadata.created_at = datetime.now().isoformat()
        self.logger.info("Initialized new FAISS index")

    def add_documents(self, pdf_paths: List[str], force_reindex: bool = False) -> Dict[str, Any]:
        """
        Add documents to the index

        Args:
            pdf_paths (List[str]): List of PDF file paths
            force_reindex (bool): Force reindexing even if document hasn't changed

        Returns:
            Dict[str, Any]: Processing results
        """
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'new_chunks': 0,
            'total_chunks': len(self.chunk_metadata)
        }

        for pdf_path in tqdm(pdf_paths, desc="Processing documents"):
            try:
                current_hash = generate_file_hash(pdf_path)
                document_id = Path(pdf_path).stem

                if not force_reindex and document_id in self.index_metadata.document_hashes:
                    if self.index_metadata.document_hashes[document_id] == current_hash:
                        self.logger.info(f"Skipping unchanged document: {pdf_path}")
                        results['skipped'] += 1
                        continue

                self.logger.info(f"Processing document: {pdf_path}")
                document = parse_pdf(pdf_path)
                new_chunks = self._process_document(document, pdf_path)
                self.index_metadata.document_hashes[document_id] = current_hash
                results['processed'] += 1
                results['new_chunks'] += len(new_chunks)

            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path}: {e}")
                results['failed'] += 1
                continue

        self.index_metadata.total_documents = len(self.index_metadata.document_hashes)
        self.index_metadata.total_chunks = len(self.chunk_metadata)
        self.index_metadata.last_updated = datetime.now().isoformat()
        self._save_index()
        results['total_chunks'] = len(self.chunk_metadata)
        return results

    def _process_document(self, document: Dict, file_path: str) -> List[ChunkMetadata]:
        """
        Process a document and create chunks

        Args:
            document (Dict): Parsed document
            file_path (str): Path to original PDF file

        Returns:
            List[ChunkMetadata]: Created chunk metadata
        """
        document_id = Path(file_path).stem
        new_chunks = []
        self._remove_document_chunks(document_id)
        chunks_to_embed: List[Tuple[str, ChunkMetadata]] = []
        document_chunks_text: Dict[str, str] = {}

        # 1. Abstract (if available)
        abstract = None
        if 'metadata' in document and document['metadata']:
            abstract = document['metadata'].get('abstract')
        if not abstract and 'sections' in document:
            for section in document['sections']:
                if section.get('heading', '').lower().startswith('abstract'):
                    abstract = section.get('body')
                    break
        if abstract:
            chunk_id = f"{document_id}_abstract"
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document.get('metadata', {}).get('title', document_id),
                section_title="Abstract",
                chunk_index=0,
                start_char=0,
                end_char=len(abstract),
                word_count=len(abstract.split()),
                chunk_type="abstract"
            )
            chunks_to_embed.append((abstract, chunk_metadata))
            new_chunks.append(chunk_metadata)
            document_chunks_text[chunk_id] = abstract

        # 2. Main text chunks
        if 'chunks' in document:
            for i, chunk_data in enumerate(document['chunks']):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_text = chunk_data['text']
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_title=document.get('metadata', {}).get('title', document_id),
                    section_title=None,  # Could be improved if section info is available
                    chunk_index=i,
                    start_char=chunk_data.get('start_char', 0),
                    end_char=chunk_data.get('end_char', 0),
                    word_count=len(chunk_text.split()),
                    chunk_type="text",
                    section_type=None
                )
                chunks_to_embed.append((chunk_text, chunk_metadata))
                new_chunks.append(chunk_metadata)
                document_chunks_text[chunk_id] = chunk_text

        # 3. Table content
        if 'tables' in document:
            for i, table in enumerate(document['tables']):
                table_text = table.get('markdown') or ''
                if table_text:
                    chunk_id = f"{document_id}_table_{i}"
                    chunk_metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        document_title=document.get('metadata', {}).get('title', document_id),
                        section_title=f"Table {i + 1}",
                        chunk_index=len(new_chunks),
                        start_char=0,
                        end_char=len(table_text),
                        word_count=len(table_text.split()),
                        chunk_type="table",
                        page_number=None
                    )
                    chunks_to_embed.append((table_text, chunk_metadata))
                    new_chunks.append(chunk_metadata)
                    document_chunks_text[chunk_id] = table_text

        # 4. Figure captions
        if 'figures' in document:
            for i, figure in enumerate(document['figures']):
                caption_text = figure.get('caption', '')
                if caption_text:
                    chunk_id = f"{document_id}_figure_{i}"
                    chunk_metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        document_title=document.get('metadata', {}).get('title', document_id),
                        section_title=f"Figure {i + 1}",
                        chunk_index=len(new_chunks),
                        start_char=0,
                        end_char=len(caption_text),
                        word_count=len(caption_text.split()),
                        chunk_type="figure",
                        page_number=None
                    )
                    chunks_to_embed.append((caption_text, chunk_metadata))
                    new_chunks.append(chunk_metadata)
                    document_chunks_text[chunk_id] = caption_text

        # Add new chunk texts to the main storage
        self.chunk_texts.update(document_chunks_text)

        if chunks_to_embed:
            self._add_chunks_to_index(chunks_to_embed)

        return new_chunks

    def _remove_document_chunks(self, document_id: str):
        """Remove existing chunks and their texts for a document"""
        # Get indices and chunk IDs of chunks to remove
        indices_to_remove = []
        chunk_ids_to_remove = []
        new_chunk_metadata = []

        for i, chunk in enumerate(self.chunk_metadata):
            if chunk.document_id == document_id:
                indices_to_remove.append(i)
                chunk_ids_to_remove.append(chunk.chunk_id)
            else:
                new_chunk_metadata.append(chunk)

        if indices_to_remove:
            # Remove chunk texts
            for chunk_id in chunk_ids_to_remove:
                self.chunk_texts.pop(chunk_id, None)

            # Rebuild index without removed chunks
            self.logger.info(f"Removing {len(indices_to_remove)} existing chunks for {document_id}")
            self._rebuild_index(new_chunk_metadata)

    def _rebuild_index(self, new_chunk_metadata: List[ChunkMetadata]):
        """Rebuild the FAISS index with filtered chunks"""
        if not new_chunk_metadata:
            self._initialize_new_index()
            return

        # Get embeddings for remaining chunks
        remaining_texts = []
        for chunk in new_chunk_metadata:
            # This is a simplified approach - in a production system,
            # you'd want to cache embeddings to avoid recomputation
            remaining_texts.append(chunk.chunk_id)  # Placeholder

        # For now, we'll initialize a new index and let it be rebuilt
        # In production, you'd want to implement proper index management
        self._initialize_new_index()
        self.chunk_metadata = new_chunk_metadata

    def _add_chunks_to_index(self, chunks_to_embed: List[Tuple[str, ChunkMetadata]]):
        """Add chunks to the FAISS index"""
        if not chunks_to_embed:
            return

        # Extract texts for embedding
        texts = [chunk[0] for chunk in chunks_to_embed]
        metadatas = [chunk[1] for chunk in chunks_to_embed]

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Add metadata
        self.chunk_metadata.extend(metadatas)

        self.logger.info(f"Added {len(texts)} chunks to index")

    def _get_section_for_chunk(self, document: Dict, chunk_data: Dict) -> Optional[str]:
        """Determine which section a chunk belongs to"""
        start_char = chunk_data.get('start_char', 0)

        for section in document.get('sections', []):
            if section.get('start_pos', 0) <= start_char <= section.get('end_pos', 0):
                return section.get('title', 'Unknown Section')

        return None

    def _detect_section_type(self, text: str) -> Optional[str]:
        """Detect the type of section based on content"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['abstract', 'summary']):
            return 'abstract'
        elif any(keyword in text_lower for keyword in ['introduction', 'background']):
            return 'introduction'
        elif any(keyword in text_lower for keyword in ['method', 'approach', 'algorithm']):
            return 'methodology'
        elif any(keyword in text_lower for keyword in ['result', 'experiment', 'evaluation']):
            return 'results'
        elif any(keyword in text_lower for keyword in ['discussion', 'analysis']):
            return 'discussion'
        elif any(keyword in text_lower for keyword in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(keyword in text_lower for keyword in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'content'

    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to searchable text"""
        if not table or not table.get('data'):
            return ""

        # Create searchable text from table
        text_parts = []

        # Add headers
        headers = table.get('headers', [])
        if headers:
            text_parts.append(f"Table headers: {', '.join(headers)}")

        # Add data rows
        for row in table.get('data', []):
            if isinstance(row, dict):
                row_text = ', '.join(f"{k}: {v}" for k, v in row.items() if v)
                text_parts.append(row_text)
            elif isinstance(row, list):
                row_text = ', '.join(str(cell) for cell in row if cell)
                text_parts.append(row_text)

        return '\n'.join(text_parts)

    def search(self, query: str, top_k: int = 5, filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search the index for relevant chunks

        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            filter_params (Dict): Optional filters (document_type, section_type, etc.)

        Returns:
            List[Dict[str, Any]]: Search results with metadata
        """
        if self.index is None or len(self.chunk_metadata) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 2)  # Get more for filtering

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk_meta = self.chunk_metadata[idx]

                # Apply filters if provided
                if filter_params and not self._passes_filters(chunk_meta, filter_params):
                    continue

                result = {
                    'chunk_metadata': chunk_meta.to_dict(),
                    'similarity_score': float(score),
                    'chunk_id': chunk_meta.chunk_id,
                    'document_id': chunk_meta.document_id,
                    'document_title': chunk_meta.document_title,
                    'section_title': chunk_meta.section_title,
                    'chunk_type': chunk_meta.chunk_type
                }
                results.append(result)

                if len(results) >= top_k:
                    break

        return results

    def _passes_filters(self, chunk_meta: ChunkMetadata, filter_params: Dict[str, Any]) -> bool:
        """Check if chunk passes the given filters"""
        for key, value in filter_params.items():
            if hasattr(chunk_meta, key):
                if getattr(chunk_meta, key) != value:
                    return False
            elif key == 'document_types' and chunk_meta.chunk_type not in value:
                return False
            elif key == 'section_types' and chunk_meta.section_type not in value:
                return False

        return True

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """
        Retrieve the full text of a chunk by its ID from storage.
        """
        return self.chunk_texts.get(chunk_id)

    def _save_index(self):
        """Save the index, metadata, and chunk texts to disk"""
        try:
            # Save FAISS index
            index_path = self.index_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = self.index_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.index_metadata), f, indent=2)

            # Save chunk metadata
            chunks_metadata_path = self.index_dir / "chunks_metadata.json"
            with open(chunks_metadata_path, 'w', encoding='utf-8') as f:
                json.dump([chunk.to_dict() for chunk in self.chunk_metadata], f, indent=2)

            # Save chunk texts
            chunks_text_path = self.index_dir / "chunks_text.json"
            with open(chunks_text_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_texts, f, indent=2)

            self.logger.info(f"Saved index with {len(self.chunk_metadata)} chunks and {len(self.chunk_texts)} chunk texts to {self.index_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        stats = {
            'total_chunks': len(self.chunk_metadata),
            'total_documents': self.index_metadata.total_documents,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model': self.model_name,
            'created_at': self.index_metadata.created_at,
            'last_updated': self.index_metadata.last_updated,
            'chunk_types': {},
            'section_types': {},
            'documents': list(self.index_metadata.document_hashes.keys())
        }

        # Count chunk types
        for chunk in self.chunk_metadata:
            chunk_type = chunk.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1

            if chunk.section_type:
                section_type = chunk.section_type
                stats['section_types'][section_type] = stats['section_types'].get(section_type, 0) + 1

        return stats

    def optimize_index(self):
        """Optimize the FAISS index for better performance"""
        if self.index is None or len(self.chunk_metadata) == 0:
            return

        self.logger.info("Optimizing FAISS index...")

        # For larger indices, you might want to use IndexIVFFlat or IndexHNSW
        # For now, we'll keep the simple IndexFlatIP for accuracy

        # You could implement index optimization here based on size
        current_size = len(self.chunk_metadata)

        if current_size > 10000:  # For large indices
            # Convert to IVF index for faster search
            nlist = min(100, current_size // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)

            # Train the index
            all_vectors = np.array([self.index.reconstruct(i) for i in range(current_size)])
            new_index.train(all_vectors.astype(np.float32))
            new_index.add(all_vectors.astype(np.float32))

            self.index = new_index
            self.logger.info(f"Optimized index with IVF structure ({nlist} clusters)")

        self._save_index()


def build_index_from_directory(pdf_directory: str,
                               index_dir: str = "./index",
                               model_name: str = "all-MiniLM-L6-v2",
                               force_rebuild: bool = False) -> VectorIndexBuilder:
    """
    Build index from all PDFs in a directory

    Args:
        pdf_directory (str): Directory containing PDF files
        index_dir (str): Directory to save index
        model_name (str): Embedding model name
        force_rebuild (bool): Force rebuild even if index exists

    Returns:
        VectorIndexBuilder: Built index
    """
    # Find all PDF files
    pdf_path = Path(pdf_directory)
    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")

    # Initialize builder
    builder = VectorIndexBuilder(
        model_name=model_name,
        index_dir=index_dir
    )

    # Build index
    results = builder.add_documents([str(f) for f in pdf_files], force_reindex=force_rebuild)

    # Optimize index
    builder.optimize_index()

    print(f"Index building complete:")
    print(f"  Processed: {results['processed']} documents")
    print(f"  Skipped: {results['skipped']} documents")
    print(f"  Failed: {results['failed']} documents")
    print(f"  Total chunks: {results['total_chunks']}")

    return builder