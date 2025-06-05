"""
Utility functions for the Document Q&A AI Agent.
Handles text processing, chunking, cleaning, and helper functions.
"""

import re
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import hashlib
from datetime import datetime


# =======================
# Logging Setup
# =======================
def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the application.
    Returns a logger instance.
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        fh = logging.FileHandler('doc_qa_agent.log')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# =======================
# Text Cleaning & Normalization
# =======================
def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text from PDFs.
    """
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s*\.\s*\.\s*\.', '...', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\(\)\[\]\-\+\=\%\$\@\#\&\*\/\\]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# =======================
# Equation Extraction
# =======================
def extract_equations(text: str) -> List[Dict[str, str]]:
    """
    Extract LaTeX-style equations from the document.
    """
    equations = []
    latex_patterns = [
        r'\$\$(.+?)\$\$',
        r'\$(.+?)\$',
        r'\\begin\{equation\}(.+?)\\end\{equation\}',
        r'\\begin\{align\}(.+?)\\end\{align\}',
        r'\\begin\{eqnarray\}(.+?)\\end\{eqnarray\}'
    ]

    for pattern in latex_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            equations.append({
                'equation': match.group(1).strip(),
                'start': match.start(),
                'end': match.end(),
                'type': 'latex'
            })
    return equations


# =======================
# Chunking
# =======================
def intelligent_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunk text into semantically meaningful units.
    """
    if not text:
        return []

    chunks = []
    section_markers = [
        r'\n\s*(?:Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References).*?\n',
        r'\n\s*\d+\.?\s+[A-Z][^.\n]*\n',
        r'\n\s*[A-Z][A-Z\s]{2,}[A-Z]\s*\n'
    ]

    sections = [text]
    for pattern in section_markers:
        new_sections = []
        for section in sections:
            splits = re.split(pattern, section, flags=re.IGNORECASE)
            new_sections.extend([s.strip() for s in splits if s.strip()])
        sections = new_sections

    for section_idx, section in enumerate(sections):
        if len(section) <= chunk_size:
            chunks.append({
                'text': section,
                'chunk_id': len(chunks),
                'section_id': section_idx,
                'start_char': 0,
                'end_char': len(section),
                'word_count': len(section.split())
            })
        else:
            section_chunks = _split_large_section(section, chunk_size, chunk_overlap)
            for chunk in section_chunks:
                chunk['section_id'] = section_idx
                chunk['chunk_id'] = len(chunks)
                chunks.append(chunk)

    return chunks


def _split_large_section(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, current_start = [], "", 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_char': current_start,
                'end_char': current_start + len(current_chunk),
                'word_count': len(current_chunk.split())
            })
            overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
            current_start += len(current_chunk) - len(overlap_text)
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start_char': current_start,
            'end_char': current_start + len(current_chunk),
            'word_count': len(current_chunk.split())
        })

    return chunks


def _get_overlap_text(text: str, overlap_size: int) -> str:
    if len(text) <= overlap_size:
        return text
    region = text[-overlap_size:]
    sentences = re.split(r'(?<=[.!?])\s+', region)
    return sentences[-1] if len(sentences) > 1 else region


# =======================
# Metadata Extraction
# =======================
def extract_metadata(text: str) -> Dict[str, Any]:
    metadata = {
        'title': None,
        'authors': [],
        'abstract': None,
        'sections': [],
        'references_count': 0,
        'equations_count': 0,
        'figures_count': 0,
        'tables_count': 0
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in lines[:6]:
            if len(line) > 10 and not re.match(r'abstract|introduction', line.lower()):
                metadata['title'] = line
                break

    match = re.search(r'abstract\s*:?\s*\n(.*?)(?=\n\s*(?:keywords|introduction|1\.|i\.|\n\n))',
                      text, re.IGNORECASE | re.DOTALL)
    if match:
        metadata['abstract'] = clean_text(match.group(1))

    metadata['equations_count'] = len(extract_equations(text))
    metadata['figures_count'] = len(re.findall(r'figure\s+\d+', text, re.IGNORECASE))
    metadata['tables_count'] = len(re.findall(r'table\s+\d+', text, re.IGNORECASE))
    metadata['references_count'] = len(re.findall(r'\[\d+\]|\(\d{4}\)', text))

    section_pattern = r'\n\s*(\d+\.?\s+[A-Z][^.\n]*)\n'
    metadata['sections'] = [s.strip() for s in re.findall(section_pattern, text, re.IGNORECASE)]

    return metadata


# =======================
# Validation, Hashing, etc.
# =======================
def validate_pdf_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith('.pdf'):
        return False
    try:
        size = os.path.getsize(file_path)
        if size == 0 or size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""


# =======================
# Output Formatting
# =======================
def format_response(answer: str, sources: List[Dict[str, Any]], confidence: Optional[float] = None) -> Dict[str, Any]:
    return {
        'answer': answer.strip(),
        'sources': [{
            'source_id': i + 1,
            'text': src.get('text', '')[:500] + "..." if len(src.get('text', '')) > 500 else src.get('text', ''),
            'chunk_id': src.get('chunk_id'),
            'section_id': src.get('section_id'),
            'similarity_score': src.get('score', 0.0)
        } for i, src in enumerate(sources)],
        'metadata': {
            'num_sources': len(sources),
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
    }


def safe_filename(filename: str) -> str:
    safe_name = re.sub(r'[^\w\s-]', '', filename)
    return re.sub(r'[-\s]+', '-', safe_name).strip('-')


# =======================
# Constants
# =======================
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 100
SUPPORTED_EXTENSIONS = ['.pdf']
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'