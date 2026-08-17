import numpy as np
from chunker import chunk_text
from embedder import generate_embeddings, model


def cosine_similarity(a, b):
    # Calculate and return cosine similarity between two 1D numpy arrays
    cosine_sim=np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosine_sim


def search(query: str, chunks: list[str], chunk_embeddings, top_k: int = 2):
    # 1. Embed the user query string
    query_embedding = model.encode([query])[0]
    # 2. Loop through chunk_embeddings and compute cosine_similarity with query vector
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((sim, i))
    # 3. Sort chunks by similarity score
    similarities.sort(reverse=True)
    # 4. Return top_k results
    return [chunks[i] for _, i in similarities[:top_k]]

if __name__ == "__main__":
    # 1. Define sample text & generate chunks + embeddings
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models "
        "by retrieving relevant information from external knowledge bases before generating responses. "
        "Vector embeddings play a crucial role in calculating semantic similarity between queries and documents."
    )
    
    # 2. Try asking a query like: "What is RAG?" or "How do vectors work?"
    query = "What is RAG?"
    # 3. Print the top matching chunk and its score!
    chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
    chunk_embeddings = generate_embeddings(chunks)
    top_chunks = search(query, chunks, chunk_embeddings, top_k=2)
    print(f"Top matching chunks for query '{query}':")
    for i, chunk in enumerate(top_chunks):
        print(f"{i+1}. {chunk}")