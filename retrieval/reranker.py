from langchain.retrievers import ContextualCompressionRetriever

from config import (
    COHERE_API_KEY,
    RERANK_TOP_N,
    USE_LOCAL_RERANK,
    LOCAL_RERANK_MODEL
)


def get_reranker(base_retriever):
    """
    Wrap base_retriever with a reranker.
    Priority:
    1. Local BGE (recommended)
    2. Cohere (if API key available)
    """

    # 🔥 OPTION 1: Local BGE Reranker (BEST for you)
    if USE_LOCAL_RERANK:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        model = HuggingFaceCrossEncoder(
            model_name=LOCAL_RERANK_MODEL  # e.g. BAAI/bge-reranker-base
        )

        compressor = CrossEncoderReranker(
            model=model,
            top_n=RERANK_TOP_N
        )

        print("  Reranker: BGE (local)")

    # ⚡ OPTION 2: Cohere (fallback if enabled)
    elif COHERE_API_KEY:
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            model="rerank-english-v3.0",
            top_n=RERANK_TOP_N
        )

        print("  Reranker: Cohere")

    # ❌ No reranker
    else:
        raise ValueError("No reranker configured")

    # 🔗 Wrap retriever
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )