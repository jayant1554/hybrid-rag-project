from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import EMBED_MODEL, TOP_K, QDRANT_URL, QDRANT_API_KEY

COLLECTION = "sebi_docs"


# ─────────────────────────────────────────────
# 🔹 Embeddings
# ─────────────────────────────────────────────
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


# ─────────────────────────────────────────────
# 🔹 Qdrant Client (Cloud)
# ─────────────────────────────────────────────
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,  # 🔥 prevents timeout issues
    )


# ─────────────────────────────────────────────
# 🔹 Create Collection (if not exists)
# ─────────────────────────────────────────────
def create_collection_if_not_exists(client):
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION not in collections:
        print(f"📦 Creating collection: {COLLECTION}")

        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=768,               # ⚠ must match embedding size
                distance=Distance.COSINE
            ),
        )


# ─────────────────────────────────────────────
# 🔹 Embed + Store
# ─────────────────────────────────────────────
def embed_and_store(chunks: list):
    embeddings = get_embeddings()
    client = get_qdrant_client()

    create_collection_if_not_exists(client)

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    store.add_documents(chunks)

    print(f"✅ Stored {len(chunks)} chunks in '{COLLECTION}'")

    return store


# ─────────────────────────────────────────────
# 🔹 Load Vector Store
# ─────────────────────────────────────────────
def load_vector_store():
    embeddings = get_embeddings()
    client = get_qdrant_client()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )


# ─────────────────────────────────────────────
# 🔹 Dense Retriever
# ─────────────────────────────────────────────
def get_dense_retriever(k: int = TOP_K):
    store = load_vector_store()

    return store.as_retriever(
        search_kwargs={"k": k}
    )