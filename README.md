
# 🚀 Aira-RAG — Privacy-First, Local Enterprise RAG System

**Aira-RAG** is a **production-oriented Retrieval-Augmented Generation (RAG) platform** designed for **secure, private, and local AI deployments**.
It enables organizations to query their internal documents using **locally hosted LLMs**, without sending data to third-party AI providers.

The system combines **document ingestion, semantic retrieval, and chat-based interaction** through a clean, enterprise-style UI.

🔗 **Live Demo (Gradio UI)**
[https://rag-private-3.onrender.com/gradio/](https://rag-private-3.onrender.com/gradio/)

---

## 🧠 Why Aira-RAG Exists

Most RAG applications today:

* Depend entirely on **cloud LLM APIs**
* Expose sensitive documents to **external vendors**
* Cannot be deployed in **air-gapped or internal networks**
* Offer no transparency into retrieval vs inference

**Aira-RAG solves this by running the entire RAG pipeline locally**, including:

* Embeddings
* Vector storage
* Retrieval logic
* LLM inference

No document data leaves the system unless explicitly configured.

---

## ✨ Core Capabilities

### 🔒 Privacy-First Architecture

* Local document ingestion and storage
* Local LLM inference using **Ollama**
* No mandatory external API calls
* Optional OpenAI fallback only via environment flags

---

### 📄 Document Ingestion Pipeline

* Upload **PDF, DOCX, TXT** files directly from the UI
* Automatic text extraction and chunking
* Persistent semantic indexing using **ChromaDB**
* Uploaded files are tracked and managed per session

---

### 🔍 Retrieval-Augmented Generation (RAG)

* Dense vector search using **MiniLM embeddings**
* Top-K semantic retrieval from **ChromaDB**
* Context-aware answer generation
* Answers are grounded strictly in retrieved documents

---

### 🧠 Dual Interaction Modes (UI-Level Control)

The UI explicitly supports **two modes of operation**:

#### 1️⃣ RAG Mode

* Answers are generated **only from uploaded documents**
* Uses retrieval + generation
* Ideal for resumes, reports, manuals, internal docs

#### 2️⃣ Search Mode

* Direct LLM interaction without document grounding
* Useful for exploratory or general-purpose queries

This separation makes system behavior **predictable and transparent**.

---

### ⚡ Transparent Performance Feedback

Each response displays real-time execution metadata, including:

* Total response time
* Retrieval latency
* LLM inference latency
* Model used (local Ollama or OpenAI fallback)

This avoids black-box behavior and reflects **production observability thinking**.

---

### 🎛️ Interactive Chat & File Controls

* Retry last query
* Undo previous interaction
* Clear conversation state
* Select and delete ingested documents

The UI is designed for **iterative document analysis**, not one-off prompts.

---

## 🏗️ System Architecture

### High-Level Flow

```
User Uploads Document
        ↓
Text Extraction & Chunking
        ↓
Embedding Generation (MiniLM)
        ↓
ChromaDB Vector Store (Persistent)
        ↓
User Question
        ↓
Semantic Retrieval (Top-K)
        ↓
Prompt Construction (Context + Query)
        ↓
Local LLM Inference (Ollama)
        ↓
Answer + Timing Metadata
```

---

## 🧱 Technology Stack

| Layer           | Technology                 |
| --------------- | -------------------------- |
| Backend API     | FastAPI                    |
| RAG Framework   | LangChain                  |
| Vector Database | ChromaDB                   |
| Embeddings      | all-MiniLM-L6-v2           |
| LLM Runtime     | Ollama                     |
| UI              | Gradio (customized layout) |
| Monitoring      | Prometheus                 |
| Deployment      | Docker                     |

---

## ⚡ Local Setup

### 1️⃣ Environment Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run Local LLM (Ollama)

```bash
ollama pull llama3.1
ollama serve
```

(Default: `http://localhost:11434`)

### 3️⃣ Start the Application

```bash
uvicorn app.main:app --reload --port 8000
```

* Gradio UI → [http://localhost:8000/gradio](http://localhost:8000/gradio)
* Static endpoint → [http://localhost:8000](http://localhost:8000)

---

## 🌍 Live Deployment

This project is **live deployed for demonstration purposes**:

🔗 [https://rag-private-3.onrender.com/gradio/](https://rag-private-3.onrender.com/gradio/)

> ⚠️ In real enterprise usage, Aira-RAG is intended to run **inside private infrastructure**, with Ollama hosted locally on internal machines or LAN servers.

---

## 🔌 API Reference

### 📤 Upload Documents

**POST** `/upload`

```json
files[]: PDF | DOCX | TXT
collection: optional
```

---

### ❓ Ask a Question

**POST** `/ask`

```json
{
  "collection": "docs",
  "question": "What does this document say?",
  "k": 4,
  "temperature": 0.2
}
```

---

### 🧠 OpenAI-Compatible Endpoints

* `POST /v1/embeddings`
* `POST /v1/chat/completions`

  * Supports `rag_collection`
  * Returns answers with source references

This allows existing OpenAI-based applications to **switch to Aira-RAG without code changes**.

---

### 📊 Metrics

* `GET /metrics` (Prometheus)

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

Starts:

* FastAPI application (`:8000`)
* Ollama LLM service (`:11434`)
* Persistent ChromaDB storage

---

## ⚙️ Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
PREFER_OPENAI=false
OPENAI_API_KEY=optional
RAG_DEFAULT_COLLECTION=default
```

---

## 🎯 Real-World Use Cases

* Enterprise internal knowledge assistants
* Legal / healthcare document analysis
* Secure Q&A for confidential data
* LAN-only AI assistants
* OpenAI-free RAG deployments

---

## 📈 What This Project Demonstrates (Recruiter View)

* End-to-end **RAG system design**
* Local LLM orchestration with Ollama
* Semantic vector search at scale
* OpenAI-compatible API engineering
* Privacy-preserving AI architecture
* Production-focused UI and observability

This is **not a toy project** — it reflects how real RAG systems are built in industry.

---

## 🧑‍💻 Author

**Manoj Reddy**
Final-year AI & Data Science Engineer
Focused on **LLMs, RAG systems, and applied AI engineering**

---


