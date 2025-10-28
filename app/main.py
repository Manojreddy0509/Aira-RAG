import os
import uuid
import time
import logging
from typing import List, Optional, Dict, Any

import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import logging config first
from . import logging_config

# New pipeline imports
from .pipeline import query_document, ingest_documents as pipeline_ingest
from .vector_store import ensure_vector_directory, get_embeddings, sanitize_collection_name

# Legacy imports for backward compatibility
from .rag import ingest_files, retrieve_context, ensure_vectorstore
from .llm import generate_answer
from .metrics import setup_metrics, track_latency, REQUEST_COUNTER

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    collection: str
    question: str
    k: int = 4
    temperature: float = 0.2
    use_reranking: bool = False
    use_new_pipeline: bool = True  # Use new pipeline by default


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIChatMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    rag_collection: Optional[str] = None


class OpenAIEmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]


app = FastAPI(title="IntelliDoc QA (Simple RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure vector store directory exists
ensure_vector_directory()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
@track_latency("upload")
async def upload(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None),
    use_new_pipeline: bool = Form(True)
):
    """Upload and ingest documents"""
    REQUEST_COUNTER.labels(endpoint="/upload").inc()
    coll = sanitize_collection_name(collection or f"collection-{uuid.uuid4().hex[:8]}")
    
    # Save uploaded files to temp directory
    paths = []
    for uf in files:
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        dest_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_{uf.filename}")
        with open(dest_path, "wb") as out:
            out.write(await uf.read())
        paths.append(dest_path)
    
    # Ingest using new or legacy pipeline
    if use_new_pipeline:
        logger.info(f"Using new pipeline for ingestion: {len(paths)} files")
        result = pipeline_ingest(coll, paths)
        return {
            "collection": coll,
            "chunks": result["chunks"],
            "files": result["files"],
            "latency": result.get("latency", 0)
        }
    else:
        logger.info(f"Using legacy pipeline for ingestion: {len(paths)} files")
        num_chunks = ingest_files(paths, coll)
        return {"collection": coll, "chunks": num_chunks}


@app.post("/ask")
@track_latency("ask")
def ask(body: AskRequest):
    """Query documents with RAG"""
    REQUEST_COUNTER.labels(endpoint="/ask").inc()
    
    # Use new pipeline if requested
    if body.use_new_pipeline:
        logger.info(f"Using new pipeline for query: {body.question[:100]}...")
        result = query_document(
            collection=body.collection,
            question=body.question,
            top_k=body.k,
            temperature=body.temperature,
            use_reranking=body.use_reranking
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "model_info": result.get("model_info", {}),
            "latency_breakdown": result.get("latency_breakdown", {})
        }
    else:
        # Legacy pipeline
        logger.info(f"Using legacy pipeline for query: {body.question[:100]}...")
        context_docs = retrieve_context(body.collection, body.question, k=body.k)
        context_text = "\n\n".join([d.page_content for d in context_docs])
        sources = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "score": d.metadata.get("score")}
            for d in context_docs
        ]
        answer = generate_answer(
            question=body.question,
            context=context_text,
            temperature=body.temperature,
        )
        return {"answer": answer, "sources": sources}


@app.post("/v1/embeddings")
@track_latency("openai_embeddings")
def openai_embeddings(body: OpenAIEmbeddingsRequest):
    REQUEST_COUNTER.labels(endpoint="/v1/embeddings").inc()
    embeddings = get_embeddings()
    vecs = embeddings.embed_documents(body.input)
    data = [
        {"object": "embedding", "embedding": v, "index": i}
        for i, v in enumerate(vecs)
    ]
    return {"object": "list", "data": data, "model": body.model or "hf/all-MiniLM-L6-v2"}


@app.post("/v1/chat/completions")
@track_latency("openai_chat")
def openai_chat(body: OpenAIChatRequest):
    REQUEST_COUNTER.labels(endpoint="/v1/chat/completions").inc()
    question = "\n".join([m.content for m in body.messages if m.role == "user"]) or ""
    collection = body.rag_collection or os.environ.get("RAG_DEFAULT_COLLECTION")
    context_text = ""
    sources: List[Dict[str, Any]] = []
    if collection:
        docs = retrieve_context(collection, question, k=4)
        context_text = "\n\n".join([d.page_content for d in docs])
        sources = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "score": d.metadata.get("score")}
            for d in docs
        ]
    answer = generate_answer(question=question, context=context_text, temperature=body.temperature)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model or os.environ.get("OLLAMA_MODEL", "llama3:latest"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "rag_sources": sources,
    }


# Gradio UI

def _build_gradio_interface():
    # Custom CSS for dark blue theme and compact layout
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        max-height: 100vh;
        overflow: hidden;
    }
    .compact-space {
        margin: 3px 0 !important;
        padding: 3px 0 !important;
    }
    h3 {
        margin-top: 8px !important;
        margin-bottom: 5px !important;
    }
    """
    
    with gr.Blocks(title="Aira - GPT", css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header - more compact
        gr.HTML("""
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
            <h1 style="color: white; margin: 0; font-size: 1.8em; font-weight: 600;">AIRA - GPT</h1>
        </div>
        """)
        
        with gr.Row():
            # Left Panel - Controls
            with gr.Column(scale=1, min_width=280):
                # Mode Selection
                gr.HTML('<h4 style="color: #1e3a8a; font-weight: bold; margin: 5px 0;">Mode</h4>')
                mode = gr.Radio(
                    choices=["RAG", "Search"],
                    value="RAG",
                    label="",
                    info=""
                )
                
                # File Upload (moved right below mode)
                file_uploader = gr.File(
                    label="Upload File(s)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                upload_btn = gr.Button(
                    "Upload File(s)",
                    variant="primary",
                    size="sm"
                )
                
                # Ingested Files
                gr.HTML('<h4 style="color: #1e3a8a; font-weight: bold; margin: 10px 0 5px 0;">Ingested Files</h4>')
                ingested_files = gr.Dataframe(
                    headers=["File Name"],
                    datatype=["str"],
                    row_count=2,
                    col_count=(1, "fixed"),
                    label="",
                    interactive=False
                )
                
                # File Management
                gr.HTML('<p style="color: #1e3a8a; font-size: 0.85em; margin: 5px 0;">Selected for Query or Deletion</p>')
                with gr.Row():
                    deselect_btn = gr.Button("De-select", size="sm")
                    delete_btn = gr.Button("Delete", size="sm", variant="stop")
                
                # Collection info (hidden but used for state)
                collection_state = gr.State(value="")
            
            # Right Panel - Chat Interface
            with gr.Column(scale=2):
                # Model info display
                model_display = gr.Textbox(
                    label="",
                    value="LLM: ollama | Model: llama3",
                    interactive=False,
                    show_label=False,
                    max_lines=1
                )
                
                # Chat messages - reduced height to fit page
                chatbot = gr.Chatbot(
                    label="",
                    height=400,
                    show_label=False,
                    bubble_full_width=False
                )
                
                # Input area (removed Top-K)
                question_input = gr.Textbox(
                    label="",
                    placeholder="Type a message...",
                    lines=2,
                    show_label=False,
                    max_lines=2
                )
                
                # Bottom buttons
                gr.HTML('<p style="color: #1e3a8a; font-size: 0.85em; margin: 5px 0;">Additional Inputs</p>')
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=3, size="sm")
                    retry_btn = gr.Button("Retry", scale=1, size="sm")
                    undo_btn = gr.Button("Undo", scale=1, size="sm")
                    clear_btn = gr.Button("Clear", scale=1, size="sm")

        # Function definitions
        def upload_files(files, current_collection):
            """Handle file upload"""
            if not files:
                return None, [], current_collection, "❌ Please select files to upload"
            
            # Generate or reuse collection
            if not current_collection:
                coll = sanitize_collection_name(f"collection-{uuid.uuid4().hex[:8]}")
            else:
                coll = current_collection
            
            paths = [f.name for f in files]
            
            try:
                # Ingest files
                result = pipeline_ingest(coll, paths)
                
                # Prepare file list for display
                file_names = [[fname] for fname in result.get('files', [])]
                
                # Include latency in status line
                latency = result.get('latency', 0.0)
                status = (
                    f"✅ Uploaded {len(result.get('files', []))} file(s) | "
                    f"{result['chunks']} chunks | {latency:.2f}s"
                )
                
                # Clear file uploader by returning None, then return other outputs
                return None, file_names, coll, status
            except Exception as e:
                return None, [], coll, f"❌ Error: {str(e)}"
        
        def ask_question(mode, collection, question, chat_history):
            """Handle question submission"""
            if not collection:
                return chat_history + [[
                    question,
                    "⚠️ Please upload documents first before asking questions."
                ]], "", ""
            
            if not question.strip():
                return chat_history, question, ""
            
            try:
                model_text = ""
                if mode == "RAG":
                    # Use RAG pipeline (fixed top_k=4)
                    result = query_document(
                        collection=collection,
                        question=question,
                        top_k=4,
                        use_reranking=True
                    )
                    
                    # Answer already includes sources from pipeline
                    answer = result["answer"]
                    lb = result.get("latency_breakdown", {})
                    total = lb.get("total", 0.0)
                    retrieval = lb.get("retrieval", 0.0)
                    llm = lb.get("llm_call", 0.0)
                    mi = result.get("model_info", {})
                    model = mi.get("model", "?")
                    provider = mi.get("provider", "?")
                    model_text = f"Answered in {total:.2f}s | retrieval {retrieval:.2f}s | llm {llm:.2f}s | {provider}:{model}"
                    
                else:  # Search mode
                    # Just retrieve and show relevant chunks
                    from .vector_store import VectorStore
                    vs = VectorStore(collection)
                    results = vs.retrieve(question, top_k=4)
                    
                    answer = "**🔍 Search Results:**\n\n"
                    for idx, res in enumerate(results, 1):
                        answer += f"**{idx}. [{res.document.metadata.get('source', 'Unknown')} - Page {res.document.metadata.get('page', 'N/A')}]**\n"
                        answer += f"Score: {res.score:.3f}\n"
                        answer += f"{res.document.page_content[:300]}...\n\n"
                    model_text = f"Search completed in N/A"
                
                # Update chat history
                chat_history = chat_history + [[question, answer]]
                
                return chat_history, "", model_text
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return chat_history + [[question, error_msg]], "", ""
        
        def clear_chat():
            """Clear chat history"""
            return [], ""
        
        def deselect_files():
            """Deselect files"""
            return []
        
        # Event handlers
        upload_btn.click(
            upload_files,
            inputs=[file_uploader, collection_state],
            outputs=[file_uploader, ingested_files, collection_state, model_display]
        )
        
        submit_btn.click(
            ask_question,
            inputs=[mode, collection_state, question_input, chatbot],
            outputs=[chatbot, question_input, model_display]
        )
        
        question_input.submit(
            ask_question,
            inputs=[mode, collection_state, question_input, chatbot],
            outputs=[chatbot, question_input, model_display]
        )
        
        retry_btn.click(
            lambda: None,
            outputs=[]
        )
        
        undo_btn.click(
            lambda hist: hist[:-1] if hist else hist,
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, question_input]
        )
        
        deselect_btn.click(
            deselect_files,
            outputs=[ingested_files]
        )
    
    return demo


demo = _build_gradio_interface()
app = gr.mount_gradio_app(app, demo, path="/gradio")

setup_metrics(app)
