import os
import streamlit as st
from pypdf import PdfReader
from google import genai

from chunker import chunk_text
from embedder import generate_embeddings
from vector_store import search

# Page Configuration
st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Knowledge Assistant")
st.caption("Upload your documents, ask questions, and get cited answers.")

# Sidebar: API Key & Document Upload
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    st.markdown("---")
    
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)
    process_btn = st.button("Process & Index Documents", type="primary")

# Initialize Session State
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process Uploaded Files
if process_btn and uploaded_files:
    all_chunks = []
    with st.spinner("Parsing and chunking documents..."):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8")
                all_chunks.extend(chunk_text(text, source=file_name, page=1))
            elif file_name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                for page_idx, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        all_chunks.extend(chunk_text(page_text, source=file_name, page=page_idx))

    if all_chunks:
        with st.spinner("Generating 384D vector embeddings..."):
            texts_to_embed = [c["text"] for c in all_chunks]
            embeddings = generate_embeddings(texts_to_embed)
            
            st.session_state.chunks = all_chunks
            st.session_state.chunk_embeddings = embeddings
            st.success(f"Indexed {len(all_chunks)} chunks across {len(uploaded_files)} document(s)!")
    else:
        st.warning("No readable text found in the uploaded documents.")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources & Citations"):
                for src in message["sources"]:
                    st.markdown(f"**File:** `{src['source']}` | **Page:** {src['page']} | **Similarity:** `{src['score']}`")
                    st.caption(f"> \"{src['text']}\"")

# Chat Input & RAG Pipeline
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not st.session_state.chunks:
        st.error("Please upload and process at least one document first.")
    else:
        # 1. Display user query
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Retrieve Top Chunks
        top_chunks = search(user_query, st.session_state.chunks, st.session_state.chunk_embeddings, top_k=3)
        
        # 3. Construct Augmented Prompt
        context_str = "\n---\n".join(
            [f"Source: {c['source']} (Page {c['page']})\nContent: {c['text']}" for c in top_chunks]
        )

        prompt = f"""You are a factual AI assistant. Answer the question using ONLY the provided context.
If the answer is not contained in the context, state that the documents do not contain enough information.

Context:
{context_str}

Question: {user_query}
Answer:"""

        # 4. Generate Answer via Google GenAI SDK
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
                answer_text = response.text
                st.markdown(answer_text)

                # 5. Display Source Citations
                with st.expander("🔍 View Sources & Citations"):
                    for src in top_chunks:
                        st.markdown(f"**File:** `{src['source']}` | **Page:** {src['page']} | **Similarity:** `{src['score']}`")
                        st.caption(f"> \"{src['text']}\"")

        # Save assistant message in state
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "sources": top_chunks
        })