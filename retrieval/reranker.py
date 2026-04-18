from langchain.retrievers import ContextualCompressionRetriever
from config import (
    RERANK_TOP_N,
    USE_LOCAL_RERANK,
    LOCAL_RERANK_MODEL
)
import logging

logger = logging.getLogger(__name__)

def get_reranker(base_retriever):
    """
    Wrap base_retriever with reranking.
    
    ✅ Options:
    1. Local BGE (FREE - no API costs)
    2. No reranking (still works great!)
    
    ❌ Removed:
    - Cohere (hits rate limits quickly)
    - Any API-based reranking
    
    Result: Zero API costs, 100% local control
    """
    
    # ─────────────────────────────────────────
    # OPTION 1: Local BGE (Completely FREE)
    # ─────────────────────────────────────────
    if USE_LOCAL_RERANK:
        try:
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            
            logger.info(f"🤖 Loading BGE: {LOCAL_RERANK_MODEL}")
            
            model = HuggingFaceCrossEncoder(
                model_name=LOCAL_RERANK_MODEL
            )
            
            compressor = CrossEncoderReranker(
                model=model,
                top_n=RERANK_TOP_N
            )
            
            print(f"✅ Reranker: Local BGE ({LOCAL_RERANK_MODEL})")
            print(f"   Top-N: {RERANK_TOP_N} documents")
            
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
        except Exception as error:
            error_msg = str(error)
            
            # Network error (common in India)
            if any(x in error_msg for x in ["getaddrinfo", "Failed to establish", "Connection", "timeout"]):
                logger.warning("⚠️ BGE download failed (network issue)")
                print("⚠️ BGE model download failed - network unreachable")
                print("   Falling back to hybrid search (no reranking)")
            
            # Other errors
            else:
                logger.error(f"❌ BGE Load Error: {error_msg}")
                print(f"❌ BGE Error: {error_msg}")
                print("   Falling back to hybrid search (no reranking)")
    
    # ─────────────────────────────────────────
    # FALLBACK: No Reranking (Still Excellent!)
    # ─────────────────────────────────────────
    logger.warning("ℹ️ Using base retriever (no reranking)")
    print("ℹ️ Hybrid search without reranking (still provides great results!)")
    
    return base_retriever