"""
Remove duplicate PDFs based on file hash
"""

import hashlib
from pathlib import Path
from logger import setup_logger

logger = setup_logger("dedup")

DATA_DIR = Path("data/processed/filtered")


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = list(DATA_DIR.glob("*.pdf"))

    seen = {}
    removed = 0

    for file in files:
        h = file_hash(file)

        if h in seen:
            logger.warning(f"Duplicate removed: {file.name}")
            file.unlink()
            removed += 1
        else:
            seen[h] = file

    logger.info(f"Removed {removed} duplicates")


if __name__ == "__main__":
    main()