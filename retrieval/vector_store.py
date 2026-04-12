from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import EMBED_MODEL, VECTOR_DB_DIR, TOP_K

COLLECTION = "sebi_docs"

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)

def get_qdrant_client():
    return QdrantClient(path=VECTOR_DB_DIR)

def embed_and_store(chunks: list):
    embeddings = get_embeddings()
    client     = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    store.add_documents(chunks)
    print(f"  Stored {len(chunks)} chunks in Qdrant collection '{COLLECTION}'")
    return store

def load_vector_store():
    embeddings = get_embeddings()
    client     = get_qdrant_client()
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)

def get_dense_retriever(k: int = TOP_K):
    return load_vector_store().as_retriever(search_kwargs={"k": k})