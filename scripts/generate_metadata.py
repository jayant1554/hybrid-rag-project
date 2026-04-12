"""
Generate metadata for filtered PDFs
"""

import json
from pathlib import Path
from datetime import datetime
from logger import setup_logger

logger = setup_logger("metadata")

DATA_DIR = Path("data/processed/filtered")
OUTPUT = Path("data/processed/metadata.json")


def detect_domain(name):
    name = name.lower()

    if "mutual" in name:
        return "mutual_fund"
    elif "broker" in name:
        return "broker"
    elif "icdr" in name:
        return "icdr"
    elif "kyc" in name:
        return "kyc"
    elif "surveillance" in name:
        return "surveillance"
    return "general"


def detect_doc_type(name):
    name = name.lower()

    if "master" in name:
        return "master_circular"
    elif "circular" in name:
        return "circular"
    elif "guideline" in name:
        return "guideline"
    elif "framework" in name:
        return "framework"
    return "other"


def main():
    files = list(DATA_DIR.glob("*.pdf"))

    metadata = []

    for file in files:
        size_bytes = file.stat().st_size

        metadata.append({
            "file_name": file.name,
            "path": str(file.resolve()),
            "size_kb": size_bytes // 1024,
            "domain": detect_domain(file.name),
            "doc_type": detect_doc_type(file.name),
            "large_doc": size_bytes > 5_000_000,
            "added_on": datetime.now().isoformat(),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata created for {len(metadata)} files")


if __name__ == "__main__":
    main()