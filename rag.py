from chunker import chunk_text
from embedder import generate_embeddings
from vector_store import search
import ollama


def main():
    # 1. Define Knowledge Base
    knowledge_base = (
        "Project Phoenix is an internal autonomous drone initiative started in 2026. "
        "The lead engineer on the propulsion system is Dr. Elena Vance. "
        "The drone uses a solid-state lithium-sulfur battery capable of 45 minutes of continuous flight. "
        "All telemetry data is transmitted via an encrypted LoRaWAN protocol operating at 915 MHz."
    )

    print("--- 1. Indexing Document (One-Time Setup) ---")
    chunks = chunk_text(knowledge_base, chunk_size=30, overlap=5)
    chunk_embeddings = generate_embeddings(chunks)
    print(f"Indexed {len(chunks)} chunks successfully!")

    print("\n" + "=" * 50)
    print("🤖 Local RAG Chatbot is Ready!")
    print("Type your question below, or type 'exit' or 'q' to quit.")
    print("=" * 50)

    # 2. The Interactive Chat Loop
    while True:
        try:
            # Get question from terminal
            query = input("\nAsk a question: ").strip()

            # Check for exit commands
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # Skip empty inputs
            if not query:
                continue

            # 3. Retrieve Context
            top_chunks = search(query, chunks, chunk_embeddings, top_k=3)
            context = "\n---\n".join(top_chunks)

            # 4. Construct Prompt
            system_prompt = (
                "You are a factual assistant. Answer the user's question based directly on the provided context. "
                "Keep your answer concise and accurate."
            )
            user_message = f"""Context:
{context}

Question: {query}"""

            # 5. Generate with Local LLM
            response = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": 0.1},
            )

            # 6. Output Answer
            print(f"\nAI: {response['message']['content']}")

        except KeyboardInterrupt:
            # Handle Ctrl + C gracefully
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    main()