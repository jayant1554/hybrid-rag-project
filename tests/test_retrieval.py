# tests/test_retrieval.py
# These are unit tests — no Qdrant or Ollama needed
from langchain.schema import Document
from ingestion.chunker import chunk_documents

def test_bm25_builds_from_docs():
    from retrieval.bm25_retriever import build_bm25
    docs = [
        Document(page_content="SEBI insider trading regulations section 11", metadata={}),
        Document(page_content="Mutual fund circular disclosure requirements", metadata={}),
        Document(page_content="IPO anchor investor lock-in period 30 days", metadata={}),
    ]
    retriever = build_bm25(docs, k=2)
    results = retriever.invoke("insider trading")
    assert len(results) > 0
    assert any("insider" in r.page_content.lower() for r in results)