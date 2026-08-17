import numpy as np
from google import genai


def generate_embeddings(texts: list[str], api_key: str) -> np.ndarray:
  """Generates dense embeddings via Google GenAI API with automatic model fallback."""
  if not texts:
    return np.array([])

  client = genai.Client(api_key=api_key.strip())

  # Candidate embedding model identifiers supported across Gemini API tiers
  candidate_models = [
      "text-embedding-004",
      "models/text-embedding-004",
      "embedding-001",
      "models/embedding-001",
  ]

  selected_model = None

  # Probe for the supported model on the first text
  for model_name in candidate_models:
    try:
      response = client.models.embed_content(
          model=model_name,
          contents=texts[0],
      )
      selected_model = model_name
      break
    except Exception:
      continue

  if not selected_model:
    raise RuntimeError(
        "Could not find an active embedding model for your Gemini API key."
    )

  vectors = []
  for text in texts:
    response = client.models.embed_content(
        model=selected_model,
        contents=text,
    )

    if hasattr(response, "embedding") and response.embedding:
      vectors.append(response.embedding.values)
    elif hasattr(response, "embeddings") and response.embeddings:
      vectors.append(response.embeddings[0].values)

  return np.array(vectors, dtype=np.float32)