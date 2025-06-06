# PaperPilot: Enterprise Document Q&A AI Agent

## Overview
**PaperPilot** is an enterprise-ready Document Q&A AI Agent that enables users to upload, index, and query multiple PDF documents using state-of-the-art Large Language Models (LLMs). The system supports both local (Ollama) and cloud (Hugging Face) LLM inference, semantic search with FAISS, and a modern Streamlit UI. Bonus: Includes Arxiv API lookup for scientific paper discovery.

---

## Architecture
PaperPilot follows a modular Retrieval-Augmented Generation (RAG) architecture, designed for clarity, scalability, and flexibility. The main components and data flow are illustrated below:

```
+-----------+     +-------------+     +---------+     +---------------------+     +---------+
| User/PDFs | --> | PDF Parser  | --> | Utils   | --> | Vector Index Builder| --> | FAISS   |
+-----------+     | (pdf_parser)|     | (utils) |     | (build_index)       |     | (Index) |
                  +-------------+     +---------+     +---------------------+     +---------+
                        (Extract, Chunk, Clean)            (Embed, Index)               ^        
                                                              ^                         |        
                                                              | (Query Embed)           | (Search) 
                                                              |                         |        
+-----------+     +---------------+     +---------------+     +-----------+     +-----------+
| User Query| --> | Query Engine  | --> | Language      | <-> | Ollama/HF |     | Retrieved |
+-----------+     | (query_engine)|     | Model         |     | (APIs)    |     | Chunks    |
                  +---------------+     | (models)      |     +-----------+     +-----------+
                        (Orchestrate)       (Inference)                                ^        
                              ^                                                        | (Context)
                              |                                                        |        
                              +--------------------------------------------------------+

+-----------+     +---------------+     +--------------+
| User Query| --> | Query Engine  | --> | Arxiv API    |
+-----------+     | (query_engine)|   +--------------+
                  |    (Lookup)   |         |             
                  +---------------+         | (Results)   
                                            v             
                                     +-----------+
                                     | Streamlit |
                                     | (main.py) |
                                     +-----------+
```

**Component Breakdown:**

1.  **Streamlit UI (`main.py`)**: The user interface for uploading documents, initiating indexing, and asking questions. Also displays Q&A results and Arxiv search outcomes.
2.  **PDF Parser (`pdf_parser.py`)**: Extracts text, structure, tables, figures, and metadata from PDF files.
3.  **Utilities (`utils.py`)**: Provides helper functions for text cleaning, intelligent chunking, hashing, and logging. Used by the PDF Parser and Vector Index Builder.
4.  **Vector Index Builder (`build_index.py`)**: Processes parsed documents, chunks text, embeds chunks using a Sentence Transformer (MiniLM), and builds/manages a FAISS vector index for efficient similarity search. Stores chunk metadata and text for retrieval.
5.  **Language Model Loader (`models.py`)**: Handles the loading and management of the LLM. Implements a local-first strategy, attempting to use a local Ollama instance (Llama 2 7B by default). If Ollama is not available, it falls back to using the Hugging Face pipeline (Zephyr-7b-beta by default) for cloud inference.
6.  **Query Engine (`query_engine.py`)**: Acts as the RAG orchestrator. Takes user queries, embeds them using the Embedding Model, searches the FAISS index for relevant document chunks, retrieves the chunk text, constructs a prompt with the retrieved context, and sends the prompt to the loaded Language Model to generate an answer. Also handles the Arxiv API lookup.

---

## Features
- 📄 **Multi-PDF Upload & Parsing**: Extracts structured text, tables, figures, and metadata from PDFs.
- 🔍 **Semantic Search**: Embeds and indexes document chunks using MiniLM and FAISS for fast retrieval.
- 🧠 **LLM Q&A**: Answers user questions using Llama 2 7B (local via Ollama) or Hugging Face (cloud fallback).
- 🌐 **Arxiv API Integration**: Search for scientific papers by description directly from the UI.
- ⚡ **Automatic Model Switching**: Prefers local LLM (Ollama) for speed and privacy; falls back to Hugging Face if not available.
- 🖥️ **Streamlit UI**: Intuitive web interface for upload, search, and Q&A.
- 🛡️ **Enterprise-Ready**: Modular, secure, and easy to extend.

---

## System Requirements
- **Python**: 3.8+
- **RAM**: 16GB+ recommended
- **CPU**: 4+ cores recommended
- **GPU**: (Optional) 4GB+ VRAM for local LLM acceleration
- **Ollama**: For local LLM inference ([Ollama install guide](https://ollama.com/))

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Nalwa-Jayesh/paperpilot
cd paperpilot
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Configure `.env` for Hugging Face API
Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here  # Only needed for cloud/gated models
```

### 5. (Optional) Install and Start Ollama for Local LLM
- [Ollama install instructions](https://ollama.com/download)
- Pull the Llama 2 7B model:
  ```bash
  ollama pull llama2:7b
  ollama serve  # If not started automatically
  ```

### 6. Windows Users: Hugging Face Cache Configuration
If you're using Windows and see a warning about symlinks in the Hugging Face cache, you have two options:
1. **Enable Developer Mode** (Recommended):
   - Open Windows Settings
   - Go to Privacy & Security > For Developers
   - Turn on Developer Mode
2. **Run Python as Administrator**:
   - Right-click on your terminal/PowerShell
   - Select "Run as Administrator"
   - Then run your Python application

This is needed because Hugging Face uses symlinks for efficient caching, which requires either Developer Mode or admin privileges on Windows.

---

## Usage

### 1. Start the Streamlit App
```bash
streamlit run main.py
```

### 2. In the Web UI:
- **Upload PDFs**: Drag and drop one or more PDF files.
- **Build Index**: Click "Build Index" to parse and embed documents.
- **Ask Questions**: Enter natural language questions about your documents.
- **View Answers**: See answers, supporting context, and source metadata.
- **Arxiv Search**: Use the sidebar to search for scientific papers by description.

---

## Model Switching Logic
- **Local First**: If Ollama is running and the `llama2:7b` model is available, all Q&A is handled locally for speed and privacy.
- **Cloud Fallback**: If Ollama is not available, the app automatically uses Hugging Face's `microsotf/phi-2` via the pipeline API.
- **No manual intervention needed!**

---

## Customization
- **Change LLM Model**: Edit `models.py` to use a different Ollama or Hugging Face model.
- **Chunk Size/Overlap**: Adjust in `utils.py` or `build_index.py` for different document types.
- **UI/UX**: Modify `main.py` for custom workflows or branding.

---

## Security & Privacy
- Uploaded PDFs are processed and indexed locally.
- Local LLM inference (Ollama) ensures data never leaves your machine.
- Cloud fallback uses Hugging Face API; use with caution for sensitive data.
- API tokens are loaded from `.env` and never hardcoded.

---

## License & Acknowledgments
- **License**: [MIT](LICENSE)
- **Acknowledgments**:
  - [Hugging Face Transformers & Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)
  - [Ollama](https://ollama.com/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [PyMuPDF](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber)
  - [arXiv API](https://arxiv.org/help/api/user-manual)
