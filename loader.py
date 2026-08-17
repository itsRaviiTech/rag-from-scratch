from pathlib import Path
from pypdf import PdfReader
from chunker import chunk_text


def process_file_into_chunks(file_path: Path) -> list[dict]:
    """Reads a .txt or .pdf file and chunks it while preserving page metadata."""
    chunks = []

    if file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_text(text, source=file_path.name, page=1))

    elif file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_chunks = chunk_text(
                    page_text, source=file_path.name, page=page_idx
                )
                chunks.extend(page_chunks)

    return chunks