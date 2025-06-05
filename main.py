import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List
from build_index import VectorIndexBuilder, build_index_from_directory
from query_engine import QueryEngine
from pdf_parser import parse_pdf
import shutil
import logging

st.set_page_config(page_title="Document Q&A AI Agent", layout="wide")
logger = logging.getLogger(__name__)

# --- Sidebar: Arxiv Lookup (Bonus) ---
st.sidebar.title("Arxiv Lookup (Bonus)")
arxiv_query = st.sidebar.text_input("Describe the paper you want to find on Arxiv")
if 'arxiv_results' not in st.session_state:
    st.session_state['arxiv_results'] = None
if 'engine' not in st.session_state:
    st.session_state['engine'] = None
if arxiv_query:
    if st.sidebar.button("Search Arxiv"):
        with st.spinner("Searching arXiv..."):
            if st.session_state['engine']:
                arxiv_result = st.session_state['engine'].arxiv_lookup(arxiv_query)
                st.session_state['arxiv_results'] = arxiv_result
            else:
                st.session_state['arxiv_results'] = {"error": "Engine not initialized."}
    if st.session_state['arxiv_results']:
        arxiv_result = st.session_state['arxiv_results']
        if 'error' in arxiv_result and arxiv_result['error']:
            st.sidebar.error(f"Error: {arxiv_result['error']}")
        elif arxiv_result.get('results'):
            st.sidebar.markdown("### Top arXiv Results:")
            for paper in arxiv_result['results']:
                st.sidebar.markdown(f"**[{paper['title']}]({paper['link']})**  ")
                st.sidebar.markdown(f"*Authors:* {', '.join(paper['authors'])}")
                st.sidebar.markdown(f"*Abstract:* {paper['summary'][:400]}{'...' if len(paper['summary']) > 400 else ''}")
                st.sidebar.markdown("---")

# --- Main App ---
st.title("📄 Document Q&A AI Agent")
st.write("Upload PDFs, build an index, and ask questions about your documents.")

# --- File Upload ---
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# --- Session State for Index and Engine ---
if 'index_built' not in st.session_state:
    st.session_state['index_built'] = False
if 'index_dir' not in st.session_state:
    st.session_state['index_dir'] = tempfile.mkdtemp()
if 'engine' not in st.session_state:
    st.session_state['engine'] = None

# --- Build Index Button ---
if uploaded_files:
    st.write(f"{len(uploaded_files)} PDF(s) uploaded.")
    if st.button("Build Index"):
        # Save uploaded files to temp dir
        temp_dir = Path(st.session_state['index_dir'])
        for file in uploaded_files:
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
        # Build index
        with st.spinner("Building index and embedding documents..."):
            try:
                build_index_from_directory(str(temp_dir), index_dir=str(temp_dir))
                st.session_state['engine'] = QueryEngine(index_dir=str(temp_dir))
                st.session_state['index_built'] = True
                st.success("Index built successfully!")
            except Exception as e:
                st.error(f"Failed to build index: {e}")
                logger.error(f"Index build error: {e}")
else:
    st.info("Please upload at least one PDF to begin.")

# --- Q&A Interface ---
if st.session_state.get('index_built') and st.session_state.get('engine'):
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question about the uploaded documents:")
    if user_query:
        with st.spinner("Retrieving answer..."):
            result = st.session_state['engine'].query(user_query, top_k=5)
        st.markdown(f"### 🧠 Answer\n{result['answer']}")
        with st.expander("Show supporting context and sources"):
            st.markdown("#### Context")
            st.code(result['context'])
            st.markdown("#### Sources")
            for src in result['sources']:
                meta = src.get('chunk_metadata', {})
                st.write(f"- **Type:** {meta.get('chunk_type', 'text')}, **Section:** {meta.get('section_title', '')}, **Doc:** {meta.get('document_title', '')}")
else:
    st.info("Build the index to enable Q&A.")

# --- Cleanup on exit (optional) ---
# You may want to clean up temp files/directories when the app closes. 