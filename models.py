"""
models.py
Defines the embedding and language model interfaces for the Document Q&A AI Agent.
Uses Hugging Face transformers for embeddings and LLM.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
import os
import requests

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Loads a local MiniLM embedding model using sentence-transformers.
        """
        self.model = SentenceTransformer(model_name, device=device if device else "cpu")
        logger.info(f"Loaded embedding model '{model_name}' on device '{self.model.device}'")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

class LanguageModel:
    def __init__(self, 
                 ollama_model: str = "llama2:7b",
                 hf_model: str = "HuggingFaceH4/zephyr-7b-beta", 
                 api_token: Optional[str] = None, 
                 task: str = "text-generation"):
        """
        Tries to use local Ollama (llama2:7b) for inference. If not available, falls back to Hugging Face pipeline (HuggingFaceH4/zephyr-7b-beta).
        """
        self.use_ollama = False
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = ollama_model
        self.generator = None

        # Try Ollama first
        try:
            resp = requests.post(self.ollama_url, json={"model": ollama_model, "prompt": "Hello", "stream": False}, timeout=3)
            if resp.status_code == 200:
                self.use_ollama = True
                logger.info(f"Using local Ollama model: {ollama_model}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")

        # Fallback: Hugging Face pipeline
        if not self.use_ollama:
            if api_token is None:
                api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if api_token:
                self.generator = pipeline(task=task, model=hf_model, token=api_token)
            else:
                self.generator = pipeline(task=task, model=hf_model)
            logger.info(f"Using Hugging Face model: {hf_model}")

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a text completion using Ollama if available, otherwise Hugging Face pipeline.
        """
        if self.use_ollama:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            try:
                resp = requests.post(self.ollama_url, json=payload)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
                else:
                    logger.error(f"Ollama error: {resp.text}")
                    return f"Ollama error: {resp.text}"
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
                return f"Ollama request failed: {e}"
        else:
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            return result[0]['generated_text'][len(prompt):].strip()