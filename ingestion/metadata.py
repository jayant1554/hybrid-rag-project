from datetime import datetime
from pathlib import Path

def enrich_metadata(chunks: list) -> list:
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["file_name"]    = Path(src).name
        chunk.metadata["domain"]       = "legal"
        chunk.metadata["regulator"]    = "SEBI"
        chunk.metadata["ingested_at"]  = datetime.now().isoformat()
        chunk.metadata["char_count"]   = len(chunk.page_content)
    return chunks