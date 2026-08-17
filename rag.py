from chunker import chunk_text
from embedder import generate_embeddings
from loader import load_documents_from_directory
from vector_store import search
import ollama


def main():
    # 1. Load external files dynamically from data/
    print("--- 1. Ingesting Documents ---")
    knowledge_base = load_documents_from_directory("data")

    if not knowledge_base.strip():
        print(
            "Knowledge base is empty. Please add .txt or .pdf files to the 'data/' folder."
        )
        return

    # 2. Chunk & Embed
    print("\n--- 2. Indexing Content ---")
    chunks = chunk_text(knowledge_base, chunk_size=40, overlap=8)
    chunk_embeddings = generate_embeddings(chunks)
    print(f"Successfully indexed {len(chunks)} chunks!")

    print("\n" + "=" * 50)
    print("🤖 Local Document Q&A Bot is Ready!")
    print("Type 'exit' or 'q' to quit.")
    print("=" * 50)

    # 3. Interactive Loop
    while True:
        try:
            query = input("\nAsk a question about your documents: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            top_chunks = search(query, chunks, chunk_embeddings, top_k=3)
            context = "\n---\n".join(top_chunks)

            system_prompt = (
                "You are a factual assistant. Answer the user's question based directly on the provided context. "
                "Keep your answer concise and accurate."
            )
            user_message = f"""Context:
{context}

Question: {query}"""

            response = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": 0.1},
            )

            print(f"\nAI: {response['message']['content']}")

        except KeyboardInterrupt:
            print("\nSession closed.")
            break


if __name__ == "__main__":
    main()