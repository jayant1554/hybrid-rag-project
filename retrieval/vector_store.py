from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import EMBED_MODEL, TOP_K

COLLECTION = "sebi_docs"
PERSIST_DIRECTORY = "./chroma_db"


# ─────────────────────────────────────────────
# 🔹 Embeddings
# ─────────────────────────────────────────────
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


# ─────────────────────────────────────────────
# 🔹 Load/Create Chroma Vector Store
# ─────────────────────────────────────────────
def get_chroma_store():
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    return store


# ─────────────────────────────────────────────
# 🔹 Embed + Store
# ─────────────────────────────────────────────
def embed_and_store(chunks: list):
    store = get_chroma_store()

    store.add_documents(chunks)

    print(f"✅ Stored {len(chunks)} chunks in '{COLLECTION}'")

    return store


# ─────────────────────────────────────────────
# 🔹 Load Vector Store
# ─────────────────────────────────────────────
def load_vector_store():
    return get_chroma_store()


# ─────────────────────────────────────────────
# 🔹 Dense Retriever
# ─────────────────────────────────────────────
def get_dense_retriever(k: int = TOP_K):
    store = load_vector_store()

    return store.as_retriever(
        search_kwargs={"k": k}
    )