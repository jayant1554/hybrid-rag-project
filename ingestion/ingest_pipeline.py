from ingestion.loader import load_directory
from ingestion.chunker import chunk_documents
from ingestion.metadata import enrich_metadata
from retrieval.vector_store import embed_and_store
from retrieval.bm25_retriever import build_bm25
from config import DATA_FILTERED_DIR


def run_ingestion(directory: str = DATA_FILTERED_DIR):
    print(f"\n=== Ingestion Pipeline: {directory} ===")

    docs   = load_directory(directory)
    chunks = chunk_documents(docs)
    chunks = enrich_metadata(chunks)

    embed_and_store(chunks)
    build_bm25(chunks)

    print(f"\nIngestion complete. {len(chunks)} chunks stored.")
    return chunks


if __name__ == "__main__":
    run_ingestion()