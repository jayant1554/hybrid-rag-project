from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, shutil, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.ingest_pipeline import run_ingestion
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from rag.chain import build_chain_with_sources
from config import ALPHA

app = FastAPI(title="SEBI RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str
    alpha: float = ALPHA
    use_rerank: bool = True
    llm_provider: str = "ollama"

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

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
    finally:
        shutil.rmtree(tmpdir)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        retriever = get_hybrid_retriever(alpha=req.alpha)
        if req.use_rerank:
            retriever = get_reranker(retriever)
        chain  = build_chain_with_sources(retriever, req.llm_provider)
        result = chain.invoke(req.question)
        sources = [{"file": d.metadata.get("file_name","?"), "page": d.metadata.get("page","?"), "excerpt": d.page_content[:300]} for d in result["sources"]]
        return QueryResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
