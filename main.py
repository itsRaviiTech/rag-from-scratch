import io
import os
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import numpy as np
from pydantic import BaseModel
from pypdf import PdfReader

from chunker import chunk_text
from embedder import generate_embeddings
from vector_store import search

app = FastAPI(title="RAG Engine REST API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_STORE = {
    "chunks": [],
    "embeddings": np.array([]),
}


class QueryRequest(BaseModel):
  query: str
  top_k: int = 3


@app.get("/")
def health_check():
  return {
      "status": "healthy",
      "indexed_chunks": len(VECTOR_STORE["chunks"]),
      "engine": "cloud-embedding-rag",
  }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), x_gemini_key: str = Header(None)
):
  if not x_gemini_key:
    raise HTTPException(
        status_code=401,
        detail="Missing X-Gemini-Key header. Provide a Gemini API key.",
    )

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
          extracted_chunks.extend(
              chunk_text(page_text, source=filename, page=page_idx)
          )
    else:
      raise HTTPException(
          status_code=400, detail="Only .txt and .pdf files are supported."
      )

    if not extracted_chunks:
      raise HTTPException(
          status_code=400, detail="No readable text found in file."
      )

    # Cloud embeddings via Gemini API (uses ~0 MB server RAM)
    texts_to_embed = [c["text"] for c in extracted_chunks]
    new_embeddings = generate_embeddings(
        texts_to_embed, api_key=x_gemini_key.strip()
    )

    if len(VECTOR_STORE["chunks"]) == 0:
      VECTOR_STORE["chunks"] = extracted_chunks
      VECTOR_STORE["embeddings"] = new_embeddings
    else:
      VECTOR_STORE["chunks"].extend(extracted_chunks)
      VECTOR_STORE["embeddings"] = np.vstack(
          [VECTOR_STORE["embeddings"], new_embeddings]
      )

    return {
        "message": f"Successfully processed '{filename}'",
        "new_chunks": len(extracted_chunks),
        "total_indexed_chunks": len(VECTOR_STORE["chunks"]),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_rag(req: QueryRequest, x_gemini_key: str = Header(None)):
  if len(VECTOR_STORE["chunks"]) == 0:
    raise HTTPException(
        status_code=400,
        detail="Vector store is empty. Please upload a document first.",
    )

  if not x_gemini_key:
    raise HTTPException(
        status_code=401,
        detail="Missing X-Gemini-Key header. Provide a Gemini API key.",
    )

  try:
    # 1. Embed query via Gemini Embedding API
    query_vector = generate_embeddings(
        [req.query], api_key=x_gemini_key.strip()
    )[0]

    # 2. In-memory Cosine Similarity
    def cosine_sim(a, b):
      return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored_chunks = []
    for i, emb in enumerate(VECTOR_STORE["embeddings"]):
      score = cosine_sim(query_vector, emb)
      scored_chunks.append((score, VECTOR_STORE["chunks"][i]))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = []
    for score, chunk in scored_chunks[: req.top_k]:
      c_copy = dict(chunk)
      c_copy["score"] = round(score, 4)
      top_chunks.append(c_copy)

    # 3. Context string with provenance
    context_str = "\n---\n".join([
        f"Source: {c['source']} (Page {c['page']})\nContent: {c['text']}"
        for c in top_chunks
    ])

    prompt = f"""You are a factual AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say "I cannot find this in the documents."

Context:
{context_str}

Question: {req.query}
Answer:"""

    # 4. Generate answer
    client = genai.Client(api_key=x_gemini_key.strip())
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )

    return {"answer": response.text, "sources": top_chunks}
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Search/Generation failed: {str(e)}"
    )