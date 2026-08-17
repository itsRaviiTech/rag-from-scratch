import numpy as np
import requests


def list_available_embedding_models(api_key: str) -> list[str]:
  """Queries Google's ListModels endpoint to discover valid embedding models on this API key."""
  key = api_key.strip()
  for version in ["v1beta", "v1"]:
    url = f"https://generativelanguage.googleapis.com/{version}/models?key={key}"
    try:
      res = requests.get(url, timeout=10)
      if res.status_code == 200:
        models = res.json().get("models", [])
        # Find all models supporting embedContent or batchEmbedContents
        embedding_models = [
            m["name"]
            for m in models
            if "embedContent" in m.get("supportedGenerationMethods", [])
            or "batchEmbedContents" in m.get("supportedGenerationMethods", [])
        ]
        if embedding_models:
          return embedding_models
    except Exception:
      continue
  return []


def generate_embeddings(texts: list[str], api_key: str) -> np.ndarray:
  """Discovers the active embedding model on the key and generates dense vectors."""
  if not texts:
    return np.array([])

  key = api_key.strip()

  # 1. Discover available embedding models on this key
  available_models = list_available_embedding_models(key)

  # Prioritize text-embedding-004 if available, otherwise take the first valid one
  selected_model = None
  for m in available_models:
    if "text-embedding-004" in m:
      selected_model = m
      break
  if not selected_model and available_models:
    selected_model = available_models[0]

  # Fallback candidate list if ListModels was empty
  if not selected_model:
    candidate_list = [
        "models/text-embedding-004",
        "models/embedding-001",
        "models/text-embedding-gecko",
    ]
  else:
    candidate_list = [selected_model]

  last_error = None
  for model_name in candidate_list:
    clean_model_name = (
        model_name
        if model_name.startswith("models/")
        else f"models/{model_name}"
    )

    for api_version in ["v1beta", "v1"]:
      url = f"https://generativelanguage.googleapis.com/{api_version}/{clean_model_name}:batchEmbedContents?key={key}"

      payload = {
          "requests": [
              {
                  "model": clean_model_name,
                  "content": {"parts": [{"text": t}]},
              }
              for t in texts
          ]
      }

      try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()

        if response.status_code == 200 and "embeddings" in data:
          vectors = [item["values"] for item in data["embeddings"]]
          return np.array(vectors, dtype=np.float32)
        else:
          last_error = data.get("error", {}).get("message", response.text)
      except Exception as e:
        last_error = str(e)

  # If batchEmbedContents failed, try single embedContent sequentially
  for model_name in candidate_list:
    clean_model_name = (
        model_name
        if model_name.startswith("models/")
        else f"models/{model_name}"
    )
    for api_version in ["v1beta", "v1"]:
      url = f"https://generativelanguage.googleapis.com/{api_version}/{clean_model_name}:embedContent?key={key}"
      all_vectors = []
      success = True

      for t in texts:
        payload = {
            "model": clean_model_name,
            "content": {"parts": [{"text": t}]},
        }
        try:
          response = requests.post(url, json=payload, timeout=20)
          data = response.json()
          if response.status_code == 200 and "embedding" in data:
            all_vectors.append(data["embedding"]["values"])
          else:
            success = False
            last_error = data.get("error", {}).get("message", response.text)
            break
        except Exception as e:
          success = False
          last_error = str(e)
          break

      if success and len(all_vectors) == len(texts):
        return np.array(all_vectors, dtype=np.float32)

  raise RuntimeError(
      f"Embedding failed. Available models found: {available_models}. Error:"
      f" {last_error}"
  )