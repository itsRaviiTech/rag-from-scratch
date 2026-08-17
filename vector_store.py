import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  """Calculates cosine similarity between two vectors."""
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  if norm_a == 0 or norm_b == 0:
    return 0.0
  return float(np.dot(a, b) / (norm_a * norm_b))


def search(
    query_vector: np.ndarray,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
  """Scores chunks against a precomputed query vector and returns top_k matches."""
  if not chunks or len(chunk_embeddings) == 0:
    return []

  scored_chunks = []
  for i, chunk_embedding in enumerate(chunk_embeddings):
    sim = cosine_similarity(query_vector, chunk_embedding)
    scored_chunks.append((sim, chunks[i]))

  # Sort descending by score
  scored_chunks.sort(key=lambda x: x[0], reverse=True)

  results = []
  for score, chunk in scored_chunks[:top_k]:
    chunk_with_score = dict(chunk)
    chunk_with_score["score"] = round(score, 4)
    results.append(chunk_with_score)

  return results