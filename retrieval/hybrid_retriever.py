from langchain.retrievers import EnsembleRetriever
from retrieval.vector_store import get_dense_retriever
from retrieval.bm25_retriever import load_bm25
from config import ALPHA, TOP_K

def get_hybrid_retriever(alpha: float = ALPHA, k: int = TOP_K) -> EnsembleRetriever:
    """
    alpha=0.0 → pure BM25 (keyword)
    alpha=1.0 → pure dense (semantic)
    alpha=0.4 → 60% BM25, 40% dense  (optimal for legal docs)
    """
    dense_weight  = alpha
    sparse_weight = 1 - alpha
    dense   = get_dense_retriever(k=k)
    sparse  = load_bm25(k=k)
    retriever = EnsembleRetriever(
        retrievers=[sparse, dense],
        weights=[sparse_weight, dense_weight],
    )
    print(f"  Hybrid retriever ready (BM25={sparse_weight:.1f}, dense={dense_weight:.1f})")
    return retriever
