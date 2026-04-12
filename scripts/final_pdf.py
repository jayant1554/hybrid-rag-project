"""
Filter valid PDFs (remove small/corrupt files)
"""

from pathlib import Path
import shutil
from logger import setup_logger

logger = setup_logger("filter")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/filtered")

MIN_SIZE_KB = 20


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.pdf"))
    logger.info(f"Found {len(files)} raw PDFs")

    kept = 0

    for file in files:
        size_kb = file.stat().st_size // 1024

        if size_kb < MIN_SIZE_KB:
            logger.warning(f"Removed small file: {file.name} ({size_kb} KB)")
            continue

        shutil.copy(file, OUT_DIR / file.name)
        kept += 1

    logger.info(f"Kept {kept} valid PDFs")


if __name__ == "__main__":
    main()