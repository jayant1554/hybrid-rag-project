# tests/test_metadata.py
from langchain.schema import Document
from ingestion.metadata import enrich_metadata

def test_enrich_adds_fields():
    docs = [Document(page_content="test", metadata={"source": "data/raw/master_circular_mutual_funds.pdf"})]
    enriched = enrich_metadata(docs)
    assert "file_name" in enriched[0].metadata
    assert "domain" in enriched[0].metadata
    assert "regulator" in enriched[0].metadata
    assert enriched[0].metadata["regulator"] == "SEBI"

def test_domain_detection():
    docs = [Document(page_content="test", metadata={"source": "master_circular_mutual_funds.pdf"})]
    enriched = enrich_metadata(docs)
    assert enriched[0].metadata["domain"] == "legal"