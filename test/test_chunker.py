# tests/test_chunker.py
from langchain.schema import Document
from ingestion.chunker import chunk_documents

def test_chunk_count():
    docs = [Document(page_content="word " * 1000, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1

def test_chunk_size():
    docs = [Document(page_content="word " * 1000, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 3000  # 600 tokens ≈ 2400-3000 chars

def test_metadata_preserved():
    docs = [Document(page_content="test content", metadata={"source": "sebi.pdf"})]
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["source"] == "sebi.pdf"