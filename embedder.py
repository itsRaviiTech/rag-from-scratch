import numpy as np
import requests


def generate_embeddings(texts: list[str], api_key: str) -> np.ndarray:
  """Generates 768-dimensional dense embeddings using the official Gemini REST endpoint directly.

  This avoids SDK version mismatches and runs with zero local RAM.
  """
  if not texts:
    return np.array([])

  key = api_key.strip()
  url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={key}"

  # Format batch payload as expected by Google Generative Language API
  requests_payload = [
      {
          "model": "models/text-embedding-004",
          "content": {"parts": [{"text": t}]},
      }
      for t in texts
  ]

  payload = {"requests": requests_payload}

  headers = {"Content-Type": "application/json"}

  response = requests.post(url, json=payload, headers=headers)
  data = response.json()

  if response.status_code != 200:
    error_msg = data.get("error", {}).get("message", response.text)
    raise RuntimeError(
        f"Embedding API error ({response.status_code}): {error_msg}"
    )

  if "embeddings" not in data:
    raise RuntimeError(f"Unexpected response format: {data}")

  vectors = [item["values"] for item in data["embeddings"]]
  return np.array(vectors, dtype=np.float32)