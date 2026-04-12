from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from config import DATA_FILTERED_DIR
# Supported file types
SUPPORTED = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader
}

# Load single file
def load_file(path: str) -> list:
    ext = Path(path).suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = SUPPORTED[ext](path)
    docs = loader.load()

    # Add metadata (useful for RAG)
    for doc in docs:
        doc.metadata["source"] = path

    print(f"Loaded {len(docs)} pages from {Path(path).name}")
    return docs


# Load all files in directory
def load_directory(directory: str) -> list:
    all_docs = []

    for f in Path(directory).iterdir():
        if f.suffix.lower() in SUPPORTED:
            try:
                all_docs.extend(load_file(str(f)))
            except Exception as e:
                print(f"WARNING: Could not load {f.name}: {e}")

    print(f"Total: {len(all_docs)} pages from {directory}")
    return all_docs
 