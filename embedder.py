import requests
import numpy as np

def generate_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """
    Generates dense embeddings via Google's stable v1 endpoint.
    Uses batchEmbedContents with zero server memory overhead.
    """
    if not texts:
        return np.array([])

    key = api_key.strip()
    
    # Target stable v1 endpoint (text-embedding-004 is generally available on v1)
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:batchEmbedContents?key={key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={key}",
        f"https://generativelanguage.googleapis.com/v1/models/embedding-001:batchEmbedContents?key={key}"
    ]

    last_error = None
    for url in endpoints:
        # Determine model string based on URL target
        model_name = "models/embedding-001" if "embedding-001" in url else "models/text-embedding-004"
        
        requests_payload = [
            {
                "model": model_name,
                "content": {"parts": [{"text": t}]}
            }
            for t in texts
        ]

        payload = {"requests": requests_payload}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        if response.status_code == 200 and "embeddings" in data:
            vectors = [item["values"] for item in data["embeddings"]]
            return np.array(vectors, dtype=np.float32)
        else:
            last_error = data.get("error", {}).get("message", response.text)

    raise RuntimeError(f"Embedding failed across all endpoints: {last_error}")