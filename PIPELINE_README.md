#  RAG Pipeline


## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INGESTION                                                │
│     ├─ Document Loading (PDF, DOCX, TXT)                    │
│     ├─ Metadata Extraction (doc_id, page, heading)          │
│     ├─ Chunking (recursive/semantic/fixed)                  │
│     ├─ Embedding Generation                                 │
│     └─ Vector Store Persistence (with caching)              │
│                                                              │
│  2. RETRIEVAL                                                │
│     ├─ Similarity Search                                    │
│     ├─ Score Normalization                                  │
│     └─ Optional Reranking                                   │
│                                                              │
│  3. PROMPT ORCHESTRATION                                     │
│     ├─ System Prompt (RAG/Detailed/Concise)                 │
│     ├─ Context Formatting (with metadata)                   │
│     └─ Citation Encouragement                               │
│                                                              │
│  4. LLM INVOCATION                                           │
│     ├─ Model Selection (Ollama/OpenAI)                      │
│     ├─ Fallback Logic (chat → generate)                     │
│     ├─ Streaming Support                                    │
│     └─ Error Handling                                       │
│                                                              │
│  5. RESPONSE FORMATTING                                      │
│     ├─ Answer with Citations                                │
│     ├─ Source Metadata                                      │
│     └─ Latency Metrics                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Enhanced Document Ingestion** (`ingest.py`)
- **Multi-format support**: PDF, DOCX, TXT
- **Smart chunking**: Recursive text splitting with configurable size/overlap
- **Metadata extraction**: 
  - Document ID (hash-based)
  - Page numbers
  - Section headings (heuristic detection)
  - Chunk IDs
- **Embedding caching**: Skip re-ingesting unchanged files
- **Incremental updates**: Only process new or modified documents

### 2. **VectorStore Interface** (`vector_store.py`)
- **Unified API**: Abstract interface for different vector databases
- **Batch ingestion**: Handle large document sets efficiently
- **Advanced retrieval**:
  - Similarity search with configurable top-k
  - Score normalization (L2 distance → similarity)
  - Metadata filtering
- **Reranking**: Two-stage retrieval with relevance boosting
- **Collection management**: Create, query, delete collections

### 3. **Prompt Templates** (`prompts.py`)
- **System prompts**:
  - `SYSTEM_PROMPT_RAG`: Balanced, citation-focused
  - `SYSTEM_PROMPT_DETAILED`: Comprehensive analysis
  - `SYSTEM_PROMPT_CONCISE`: Brief, focused answers
- **Context formatting**: 
  - Metadata headers (source, page, heading, relevance)
  - Structured separators
- **Source extraction**: Deduplicated citations
- **Citation formatting**: Appended source references

### 4. **LLM Abstraction** (`llm.py`)
- **Multi-provider support**:
  - Ollama (local, privacy-preserving)
  - OpenAI (fallback, cloud-based)
- **Fallback logic**: 
  1. Try OpenAI (if `PREFER_OPENAI=true`)
  2. Fall back to Ollama chat endpoint
  3. Fall back to Ollama generate endpoint
- **Streaming support**: Real-time response generation
- **Error handling**: Graceful degradation with detailed logging
- **LLMResponse object**: Structured response with metadata

### 5. **Query Pipeline** (`pipeline.py`)
- **End-to-end orchestration**: Single function for complete RAG flow
- **Configurable parameters**:
  - `top_k`: Number of chunks to retrieve
  - `temperature`: LLM creativity control
  - `use_reranking`: Enable two-stage retrieval
  - `system_prompt`: Choose prompt template
  - `max_tokens`: Control response length
- **Latency tracking**: Per-stage timing breakdown
- **Error resilience**: Graceful error handling at each stage

### 6. **Logging & Metrics** (`logging_config.py`)
- **Structured logging**:
  - Rotating file handler (10MB max, 5 backups)
  - Console output for debugging
  - Per-module log levels
- **Metrics tracking**:
  - Ingestion: document count, chunk count
  - Retrieval: query, scores, latency
  - LLM: model, provider, latency, token usage
  - Errors and fallbacks

## API Endpoints

### 1. **Upload/Ingest** (`POST /upload`)
```json
Request:
{
  "files": [<file1>, <file2>],
  "collection": "my_collection",
  "use_new_pipeline": true
}

Response:
{
  "collection": "my_collection",
  "chunks": 42,
  "files": ["doc1.pdf", "doc2.txt"],
  "latency": 3.21
}
```

### 2. **Ask/Query** (`POST /ask`)
```json
Request:
{
  "collection": "my_collection",
  "question": "What is machine learning?",
  "k": 4,
  "temperature": 0.2,
  "use_reranking": true,
  "use_new_pipeline": true
}

Response:
{
  "answer": "Machine learning is...\n\n**Sources:**\n1. ml_guide.pdf (Page 5)\n2. intro.txt (Page 1)",
  "sources": [
    {"source": "ml_guide.pdf", "page": 5, "score": 0.92, "heading": "Introduction"},
    {"source": "intro.txt", "page": 1, "score": 0.85, "heading": ""}
  ],
  "model_info": {
    "model": "llama3.1",
    "provider": "ollama",
    "temperature": 0.2
  },
  "latency_breakdown": {
    "retrieval": 0.15,
    "prompt_building": 0.01,
    "llm_call": 2.34,
    "formatting": 0.02,
    "total": 2.52
  }
}
```

### 3. **OpenAI-Compatible Endpoints**
- `POST /v1/embeddings`: Generate embeddings
- `POST /v1/chat/completions`: Chat completions with RAG

## Usage Examples

### Python API

```python
from app.pipeline import query_document, ingest_documents

# Ingest documents
result = ingest_documents(
    collection="my_docs",
    file_paths=["paper.pdf", "notes.txt"],
    chunk_size=1000,
    chunk_overlap=200,
    skip_cached=True
)
print(f"Ingested {result['chunks']} chunks from {len(result['files'])} files")

# Query documents
response = query_document(
    collection="my_docs",
    question="What are the key findings?",
    top_k=4,
    temperature=0.2,
    use_reranking=True
)
print(response["answer"])
print(f"Sources: {response['sources']}")
```

### Gradio UI

The Gradio interface at `/gradio` uses the new pipeline by default:
- **Upload**: Drag & drop PDFs, DOCX, or TXT files
- **Ask**: Enter questions and get answers with source citations
- **Reranking**: Automatically enabled for better accuracy

### CLI Tool

Run the test suite:
```bash
cd simple_rag
python tests/test_pipeline.py
```

## Configuration

### Environment Variables

```bash
# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
PREFER_OPENAI=false

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# RAG Settings (in code)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TOP_K=4
DEFAULT_TEMPERATURE=0.2
```

### Chunking Strategies

```python
from app.ingest import ChunkingStrategy

# Recursive (default): splits by paragraphs, sentences
strategy = ChunkingStrategy.RECURSIVE

# Fixed: fixed-size chunks with overlap
strategy = ChunkingStrategy.FIXED

# Semantic (future): semantic similarity-based splits
strategy = ChunkingStrategy.SEMANTIC
```

### System Prompts

```python
from app.prompts import SYSTEM_PROMPT_RAG, SYSTEM_PROMPT_DETAILED, SYSTEM_PROMPT_CONCISE

# Balanced (default)
prompt = SYSTEM_PROMPT_RAG

# Comprehensive analysis
prompt = SYSTEM_PROMPT_DETAILED

# Brief answers
prompt = SYSTEM_PROMPT_CONCISE
```

## Testing

### Run All Tests
```bash
cd simple_rag
python tests/test_pipeline.py
```

### Test Coverage
- ✓ Document chunking with metadata
- ✓ Vector store operations (add, retrieve, rerank)
- ✓ Prompt template building
- ✓ End-to-end pipeline (requires Ollama)

## Logging

### Log Files
- Location: `simple_rag/logs/simple_rag.log`
- Rotation: 10MB max, 5 backups
- Format: `YYYY-MM-DD HH:MM:SS - module - LEVEL - message`

### Log Levels
- **DEBUG**: Detailed trace for development
- **INFO**: Normal operations (default)
- **WARNING**: Unexpected events (e.g., fallbacks)
- **ERROR**: Failures requiring attention

## Performance Metrics

### Typical Latencies (on M1 Mac)
- **Ingestion**: ~500ms per PDF page
- **Embedding**: ~100ms per chunk
- **Retrieval**: ~50-150ms (depends on collection size)
- **LLM (Ollama)**: ~2-5s for 200-token answer
- **LLM (OpenAI)**: ~1-2s for 200-token answer

### Optimization Tips
1. **Increase chunk size** (1000 → 1500) to reduce chunks
2. **Enable reranking** for better relevance at cost of latency
3. **Use smaller top_k** (4 → 2) for faster retrieval
4. **Cache frequently queried results**

## Backward Compatibility

The legacy API is still supported:
```python
# Old way (still works)
from app.rag import ingest_files, retrieve_context
from app.llm import generate_answer

chunks = ingest_files(paths, collection)
docs = retrieve_context(collection, question, k=4)
answer = generate_answer(question, context)

# New way (recommended)
from app.pipeline import query_document, ingest_documents

result = ingest_documents(collection, paths)
response = query_document(collection, question)
```

Set `use_new_pipeline=false` in API requests to use legacy pipeline.

## Comparison: Old vs. New

| Feature | Old Pipeline | New Pipeline |
|---------|--------------|--------------|
| Chunking | Basic splitting | Enhanced with metadata |
| Retrieval | Simple similarity | Similarity + reranking |
| Prompts | Hardcoded | Template-based |
| LLM | Basic call | Unified with fallback |
| Logging | Minimal | Comprehensive |
| Caching | None | File-based caching |
| Streaming | No | Supported |
| Metrics | Basic | Detailed breakdown |

## Roadmap

- [ ] **Streaming UI support**: Real-time answers in Gradio
- [ ] **Conversation history**: Multi-turn dialogues
- [ ] **Cross-encoder reranking**: More accurate relevance
- [ ] **Hybrid search**: Combine dense + sparse retrieval
- [ ] **Document summarization**: Auto-generate summaries
- [ ] **Answer caching**: Cache frequent queries
- [ ] **Multi-modal support**: Images, tables, charts



- [LangChain Documentation](https://python.langchain.com/)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)

---

