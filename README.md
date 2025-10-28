# Aira-RAG

## Simple RAG Web App

A minimal, local, privacy-preserving RAG web app using FastAPI, LangChain, ChromaDB, and a locally hosted LLM (Ollama). Includes:
- PDF/DOCX/TXT ingestion
- Vector embeddings (all-MiniLM-L6-v2) and ChromaDB persistence
- RAG question answering via Ollama (optional OpenAI fallback)
- OpenAI-compatible endpoints: /v1/chat/completions and /v1/embeddings
- Gradio interface mounted at /gradio and a simple static UI at /
- Prometheus /metrics endpoint for latency and request counters
- Dockerfile + docker-compose for reproducible deployment

## Quickstart (local)

1) Python environment
- Python 3.11 recommended
- From project root, install deps:

  pip install -r requirements.txt

2) Run a local LLM (Ollama)
- Install Ollama: https://ollama.com/download
- Start the service (default at http://localhost:11434) and pull a model, e.g.:

  ollama pull llama3.1

3) Launch the app

  uvicorn app.main:app --reload --port 8000

Open http://localhost:8000 for the static UI, or http://localhost:8000/gradio for Gradio.

Environment variables (optional):
- OLLAMA_BASE_URL (default http://localhost:11434)
- OLLAMA_MODEL (default llama3.1)
- PREFER_OPENAI (default false) with OPENAI_API_KEY to use OpenAI fallback
- RAG_DEFAULT_COLLECTION (used by the OpenAI chat endpoint if no rag_collection provided)

## API

- POST /upload (multipart): fields files[]=<PDF/DOCX/TXT>, optional form field collection
  Returns: { collection, chunks }

- POST /ask (json): { collection, question, k?=4, temperature?=0.2 }
  Returns: { answer, sources: [{source, page, score}] }

- POST /v1/embeddings (OpenAI-compatible): { input: ["text", ...], model? }
  Returns: { object, data: [{embedding, index}], model }

- POST /v1/chat/completions (OpenAI-compatible, with extension):
  {
    model?,
    messages: [{role:"user", content:"..."}, ...],
    rag_collection?: "<collection>"
  }
  Returns: OpenAI chat completion shape, with rag_sources carrying citations.

- GET /metrics (Prometheus): request counters and latency histograms.

## Docker

From simple_rag/ directory:

- Build and run with Ollama:

  docker compose up --build

This starts:
- app (FastAPI on :8000)
- ollama (LLM server on :11434)

ChromaDB persistence is stored under simple_rag/chroma_db.

## Notes
- This project uses LangChain for document loading, splitting, and Chroma vector store integration.
- The LLM call defaults to Ollama for privacy. If you enable PREFER_OPENAI with OPENAI_API_KEY, the app tries OpenAI first and falls back to Ollama on error.
- For evaluation, /metrics exposes basic latency and request counts. You can extend metrics, log retrieval scores, or create an evaluation endpoint with ground-truth comparisons (e.g., cosine similarity or ROUGE).
