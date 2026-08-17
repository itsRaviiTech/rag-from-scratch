import numpy as np
from embedder import model


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """Retrieves top_k matching chunk objects with their similarity scores."""
    if not chunks or len(chunk_embeddings) == 0:
        return []

    query_embedding = model.encode([query])[0]
    scored_chunks = []

    for i, chunk_embedding in enumerate(chunk_embeddings):
        sim = float(cosine_similarity(query_embedding, chunk_embedding))
        scored_chunks.append((sim, chunks[i]))

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return top_k chunk objects with their score added
    results = []
    for score, chunk in scored_chunks[:top_k]:
        chunk_with_score = dict(chunk)
        chunk_with_score["score"] = round(score, 4)
        results.append(chunk_with_score)

    return results