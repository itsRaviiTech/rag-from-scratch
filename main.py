import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai

# Import your existing custom RAG modules
from chunker import chunk_text
from embedder import generate_embeddings
from vector_store import search

app = FastAPI(title="RAG Engine REST API", version="1.0")

# Enable CORS so your GitHub Pages site can make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (including your GitHub Pages domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document storage state (for a production microservice, this can live in a DB/VectorDB)
VECTOR_STORE = {
    "chunks": [],
    "embeddings": np.array([]),
}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
def health_check():
    """Health check route to verify the API is online."""
    return {
        "status": "healthy",
        "indexed_chunks": len(VECTOR_STORE["chunks"]),
        "engine": "custom-cosine-rag"
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    REST Endpoint to upload and parse a .txt or .pdf file,
    chunk the text with metadata, and generate vector embeddings.
    """
    file_bytes = await file.read()
    filename = file.filename
    extracted_chunks = []

    try:
        if filename.endswith(".txt"):
            text = file_bytes.decode("utf-8")
            extracted_chunks.extend(chunk_text(text, source=filename, page=1))

        elif filename.endswith(".pdf"):
            pdf_stream = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_chunks.extend(chunk_text(page_text, source=filename, page=page_idx))
        else:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

        if not extracted_chunks:
            raise HTTPException(status_code=400, detail="No readable text found in file.")

        # Generate embeddings for new chunks
        texts_to_embed = [c["text"] for c in extracted_chunks]
        new_embeddings = generate_embeddings(texts_to_embed)

        # Store in state
        if len(VECTOR_STORE["chunks"]) == 0:
            VECTOR_STORE["chunks"] = extracted_chunks
            VECTOR_STORE["embeddings"] = new_embeddings
        else:
            VECTOR_STORE["chunks"].extend(extracted_chunks)
            VECTOR_STORE["embeddings"] = np.vstack([VECTOR_STORE["embeddings"], new_embeddings])

        return {
            "message": f"Successfully processed '{filename}'",
            "new_chunks": len(extracted_chunks),
            "total_indexed_chunks": len(VECTOR_STORE["chunks"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_rag(req: QueryRequest, x_gemini_key: str = Header(None)):
    """
    REST Endpoint to perform semantic search across indexed chunks
    and generate a cited answer using the Gemini API.
    """
    if len(VECTOR_STORE["chunks"]) == 0:
        raise HTTPException(status_code=400, detail="Vector store is empty. Please upload a document first.")

    if not x_gemini_key:
        raise HTTPException(status_code=401, detail="Missing X-Gemini-Key header. Provide a Gemini API key.")

    # 1. Cosine similarity search using your custom NumPy vector_store
    top_chunks = search(
        req.query,
        VECTOR_STORE["chunks"],
        VECTOR_STORE["embeddings"],
        top_k=req.top_k
    )

    # 2. Format context with source provenance
    context_str = "\n---\n".join(
        [f"Source: {c['source']} (Page {c['page']})\nContent: {c['text']}" for c in top_chunks]
    )

    prompt = f"""You are a factual AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say "I cannot find this in the documents."

Context:
{context_str}

Question: {req.query}
Answer:"""

    # 3. Call Gemini API for grounded generation
    try:
        client = genai.Client(api_key=x_gemini_key.strip())
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return {
            "answer": response.text,
            "sources": top_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")


@app.delete("/api/clear")
def clear_vector_store():
    """Flushes the current vector store."""
    VECTOR_STORE["chunks"] = []
    VECTOR_STORE["embeddings"] = np.array([])
    return {"message": "Vector store cleared."}