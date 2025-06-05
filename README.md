# PaperPilot: Enterprise Document Q&A AI Agent

## Overview
**PaperPilot** is an enterprise-ready Document Q&A AI Agent that enables users to upload, index, and query multiple PDF documents using state-of-the-art Large Language Models (LLMs). The system supports both local (Ollama) and cloud (Hugging Face) LLM inference, semantic search with FAISS, and a modern Streamlit UI. Bonus: Includes Arxiv API lookup for scientific paper discovery.

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
git clone <your-repo-url>
cd <your-repo-directory>
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
- **Cloud Fallback**: If Ollama is not available, the app automatically uses Hugging Face's `HuggingFaceH4/zephyr-7b-beta` via the pipeline API.
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
- **License**: [MIT](LICENSE) (or your choice)
- **Acknowledgments**:
  - [Hugging Face Transformers & Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)
  - [Ollama](https://ollama.com/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [PyMuPDF](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber)
  - [arXiv API](https://arxiv.org/help/api/user-manual)
