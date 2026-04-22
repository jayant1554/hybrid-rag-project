from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, shutil, os, sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.ingest_pipeline import run_ingestion
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from rag.chain import build_chain_with_sources
from config import ALPHA

app = FastAPI(title="SEBI RAG API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Models
# -------------------------

class QueryRequest(BaseModel):
    question: str
    alpha: float = ALPHA
    use_rerank: bool = True
    llm_provider: str = "ollama"   # "ollama" | "groq" | "mistral"

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Ingestion Endpoint
# -------------------------

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    tmpdir = tempfile.mkdtemp()
    try:
        for f in files:
            path = os.path.join(tmpdir, f.filename)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)

        run_ingestion(tmpdir)

        return {"message": f"Ingested {len(files)} file(s)"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        shutil.rmtree(tmpdir)

# -------------------------
# Query Endpoint
# -------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        # Step 1: Get retriever
        retriever = get_hybrid_retriever(alpha=req.alpha)
        if retriever is None:
            raise ValueError("Retriever initialization failed")

        # Step 2: Optional reranker
        if req.use_rerank:
            retriever = get_reranker(retriever)
            if retriever is None:
                raise ValueError("Reranker failed")

        # Step 3: Build chain (IMPORTANT: pass provider STRING)
        chain = build_chain_with_sources(retriever, req.llm_provider)
        if chain is None:
            raise ValueError("Chain creation failed")

        # Step 4: Run query
        result = chain.invoke(req.question)

        # Step 5: Format sources safely
        sources = [
            {
                "file": doc.metadata.get("file_name", "?"),
                "page": doc.metadata.get("page", "?"),
                "excerpt": doc.page_content[:300]
            }
            for doc in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")