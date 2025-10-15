import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import sanitization from vector_store
from .vector_store import sanitize_collection_name

VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
VECTOR_DIR = os.path.abspath(VECTOR_DIR)

_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def ensure_vectorstore():
    os.makedirs(VECTOR_DIR, exist_ok=True)


def _load_single(path: str):
    p = path.lower()
    if p.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if p.endswith(".docx"):
        return Docx2txtLoader(path).load()
    if p.endswith(".txt"):
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {path}")


def load_documents(paths: List[str]):
    docs = []
    for p in paths:
        docs.extend(_load_single(p))
    return docs


def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)


def _get_vectorstore(collection: str) -> Chroma:
    embeddings = get_embeddings()
    # Sanitize collection name
    sanitized_collection = sanitize_collection_name(collection)
    return Chroma(embedding_function=embeddings, collection_name=sanitized_collection, persist_directory=VECTOR_DIR)


def ingest_files(paths: List[str], collection: str) -> int:
    ensure_vectorstore()
    docs = load_documents(paths)
    chunks = _split_documents(docs)
    vs = _get_vectorstore(collection)
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)


def retrieve_context(collection: str, query: str, k: int = 4):
    vs = _get_vectorstore(collection)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    for d in docs:
        if "score" not in d.metadata:
            d.metadata["score"] = d.metadata.get("distance") or None
    return docs
