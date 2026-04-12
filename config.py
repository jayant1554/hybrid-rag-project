import os 
import dotenv
dotenv.load_dotenv()
import logging
# Paths
DATA_RAW_DIR       = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_FILTERED_DIR  = "data/processed/filtered"
VECTOR_DB_DIR      = "data/vectordb"
MANIFEST_FILE      = "data/manifest.json"
PROJECT_NAME = "SEBI Regulatory Intelligence Assistant"
DOMAIN       = "legal"
authors      = "Jayant Bisht"
VERSION      = "0.1.0"
CHUNK_SIZE    = 700
CHUNK_OVERLAP  = 100
# Retrieval
ALPHA       = 0.4   # 0=keyword only  1=dense only  0.4=favor BM25 for legal
TOP_K       = 8
RERANK_TOP_N = 3
# LLM
ollama_model =os.getenv("OLLAMA_MODEL","llama-3.2")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "nomic-embed-text")

COHERE_API_KEY    = os.getenv("COHERE_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",   "")
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "sebi-hybrid-rag")
