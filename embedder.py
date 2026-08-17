# from sentence_transformers import SentenceTransformer
# from chunker import chunk_text

# # downloding the model from sentence-transformers library, this model is used to convert text into embeddings, the smaller version tho
# print("Downloading the model...")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def generate_embeddings(chunks: list[str]):
#     # model.encode takes a list of strings and returns a list of embeddings, which are numerical representations of the text
#     embeddings = model.encode(chunks)
#     return embeddings


# # testing the embedding generation with a sample text muahahaha 
# if __name__ == "__main__":
#     sample_text = (
#         "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models "
#         "by retrieving relevant information from external knowledge bases before generating responses. "
#         "Vector embeddings play a crucial role in calculating semantic similarity between queries and documents."
#     )
    
#     # 1. chunk the sample text into smaller pieces
#     # chunk_text is a function that takes a string of text and splits it into smaller chunks based on the specified chunk size and overlap
#     chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
#     print(f"generated {len(chunks)} chunks")
    
#     # 2. embed the chunks into vector embeddings
#     embeddings = generate_embeddings(chunks)
    
#     # 3. inspect the embeddings to see what they look like
#     print(f"generated {len(embeddings)} embedding vectors")
#     print(f"Dimension of each embedding vector: {len(embeddings[0])}")
#     print(f"First 5 embedding vectors:\n{embeddings[:5]}")



# Google embeddings API since its lightweigh (for resp api)
from google import genai
import numpy as np


def generate_embeddings(texts: list[str], api_key: str) -> np.ndarray:
  """Generates dense embeddings via Google GenAI API with zero local RAM usage."""
  if not texts:
    return np.array([])

  client = genai.Client(api_key=api_key.strip())

  vectors = []
  # Process embeddings via the client embedding endpoint
  for text in texts:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    # Extract the embedding values vector
    if hasattr(response, "embedding") and response.embedding:
      vectors.append(response.embedding.values)
    elif hasattr(response, "embeddings") and response.embeddings:
      vectors.append(response.embeddings[0].values)

  return np.array(vectors, dtype=np.float32)