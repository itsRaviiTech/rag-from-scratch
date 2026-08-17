def chunk_text(
    text: str,
    source: str = "unknown",
    page: int = 1,
    chunk_size: int = 40,
    overlap: int = 8,
) -> list[dict]:
    """Splits text into chunks and returns a list of dictionaries with metadata."""
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly smaller than chunk_size")

    stride = chunk_size - overlap
    chunks = []
    chunk_id = 0

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text_str,
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            }
        )
        chunk_id += 1

    return chunks