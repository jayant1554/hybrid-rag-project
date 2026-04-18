import pickle
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from config import TOP_K, DATA_PROCESSED_DIR

BM25_INDEX_PATH = Path(DATA_PROCESSED_DIR) / "bm25_index.pkl"

def build_bm25(chunks: list, k: int = TOP_K) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks, k=k)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(retriever, f)
    print(f"  BM25 index saved to {BM25_INDEX_PATH}")
    return retriever

def load_bm25(k: int = TOP_K) -> BM25Retriever:
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError("BM25 index not found. Run ingestion first.")
    with open(BM25_INDEX_PATH, "rb") as f:
        retriever = pickle.load(f)
    retriever.k = k
    return retriever
