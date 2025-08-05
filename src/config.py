"""
Configuration settings for the Document-based Chatbot application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Configuration
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = BASE_DIR / "core" / "prompts.yml"

# Server Configuration
SERVER = {
    "HOST": "127.0.0.1",
    "PORT": 8000,
    "RELOAD": True,
    "TITLE": "Document-based Chatbot",
    "DESCRIPTION": "RAG-based chatbot for documents"
}

# Logging Configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "loggers": {
        "httpx": "WARNING",
        "faiss": "WARNING"
    }
}

# LLM Configuration
LLM = {
    "primary": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "fallback": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key": os.getenv("GEMINI_API_KEY")
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
}

# Image Processing Configuration
IMAGE = {
    "max_size": 800,  # Maximum image dimension in pixels
    "quality": 85,    # JPEG compression quality
    "max_size_mb": 20,  # Maximum file size in MB
    "headers": {
        "User-Agent": "AddImageDescription/1.0 (your@email.com) PythonRequests/2.25.1"
    }
}

# Document Processing Configuration
DOCUMENT = {
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "supported_formats": [".pdf"]
}