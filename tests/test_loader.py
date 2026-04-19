from ingestion.loader import load_directory
from pathlib import Path


def test_load_directory_reads_files(tmp_path):
    # 🔹 create fake test file
    file = tmp_path / "test.txt"
    file.write_text("This is a test document.")

    docs = load_directory(str(tmp_path))

    assert len(docs) > 0
    assert "test document" in docs[0].page_content.lower()


def test_loader_adds_metadata(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Sample content")

    docs = load_directory(str(tmp_path))

    assert "source" in docs[0].metadata
    assert "sample.txt" in docs[0].metadata["source"]