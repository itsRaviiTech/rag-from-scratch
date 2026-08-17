from chunker import chunk_text
from embedder import generate_embeddings
from vector_store import search
import ollama


def main():
    # 1. Define your custom knowledge base (facts the base LLM wouldn't know)
    knowledge_base = (
        "Project Phoenix is an internal autonomous drone initiative started in 2026. "
        "The lead engineer on the propulsion system is Dr. Elena Vance. "
        "The drone uses a solid-state lithium-sulfur battery capable of 45 minutes of continuous flight. "
        "All telemetry data is transmitted via an encrypted LoRaWAN protocol operating at 915 MHz."
    )

    print("--- 1. Indexing Document ---")
    # 2. Break document into chunks
    chunks = chunk_text(knowledge_base, chunk_size=30, overlap=5)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate embeddings for all chunks
    chunk_embeddings = generate_embeddings(chunks)
    print("Generated vector embeddings.")

    # 4. Define the user's question
    query = "What battery does Project Phoenix use and who leads propulsion?"
    print(f"\n--- 2. Querying: '{query}' ---")

    # 5. Retrieve top matching chunks via Cosine Similarity
    top_chunks = search(query, chunks, chunk_embeddings, top_k=3)
    print("\n[Retrieved Context Chunks]:")
    for i, c in enumerate(top_chunks, 1):
        print(f" {i}. {c}")

    # 6. Construct Clear Roles
    context = "\n---\n".join(top_chunks)

    system_prompt = (
        "You are a factual assistant. Answer the user's question based directly on the provided context. "
        "Keep your answer concise and accurate."
    )

    user_message = f"""Context:
{context}

Question: {query}"""

    # 7. Generate Response with Structured Messages
    print("\n--- 3. Generating Response from Local LLM ---")
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        options={
            "temperature": 0.1  # Low temperature = deterministic & factual
        },
    )

    print("\nFinal Answer:")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()