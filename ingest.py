"""
Enhanced Document Ingestion Module

Provides document loading, parsing, chunking with metadata extraction,
and embedding caching for incremental ingestion.
"""
import os
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Cache directory for tracking ingested files
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "ingest_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class ChunkingStrategy:
    """Different chunking strategies for document splitting"""
    
    RECURSIVE = "recursive"  # Default: splits by paragraphs/sentences
    SEMANTIC = "semantic"    # Future: split by semantic similarity
    FIXED = "fixed"          # Fixed size chunks


def _compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file for change detection"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_cache_key(filepath: str, collection: str) -> str:
    """Generate cache key for tracking ingested files"""
    return f"{collection}_{Path(filepath).name}"


def _is_file_cached(filepath: str, collection: str) -> bool:
    """Check if file was already ingested"""
    cache_key = _get_cache_key(filepath, collection)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return False
    
    # Check if file hash matches
    try:
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        current_hash = _compute_file_hash(filepath)
        return cached_data.get("file_hash") == current_hash
    except Exception as e:
        logger.warning(f"Error reading cache for {filepath}: {e}")
        return False


def _cache_file_metadata(filepath: str, collection: str, chunk_count: int):
    """Cache metadata about ingested file"""
    cache_key = _get_cache_key(filepath, collection)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    metadata = {
        "filepath": filepath,
        "filename": Path(filepath).name,
        "file_hash": _compute_file_hash(filepath),
        "collection": collection,
        "chunk_count": chunk_count,
        "ingested_at": str(Path(filepath).stat().st_mtime)
    }
    
    with open(cache_file, "w") as f:
        json.dump(metadata, f, indent=2)


def _load_document(filepath: str) -> List[Document]:
    """Load a single document based on file type"""
    p = filepath.lower()
    
    try:
        if p.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif p.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif p.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} page(s) from {Path(filepath).name}")
        return docs
    
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def _extract_metadata(doc: Document, doc_id: str, chunk_index: int) -> Dict[str, Any]:
    """Extract and enrich document metadata"""
    metadata = doc.metadata.copy()
    
    # Add document identifier
    metadata["doc_id"] = doc_id
    metadata["chunk_id"] = f"{doc_id}_chunk_{chunk_index}"
    
    # Normalize source to filename only
    if "source" in metadata:
        metadata["source"] = Path(metadata["source"]).name
    
    # Ensure page number exists
    if "page" not in metadata:
        metadata["page"] = 0
    
    # Extract section heading from content (simple heuristic)
    content = doc.page_content.strip()
    lines = content.split("\n")
    if lines:
        first_line = lines[0].strip()
        # Consider first line as heading if it's short and doesn't end with punctuation
        if len(first_line) < 100 and not first_line.endswith((".", "!", "?")):
            metadata["heading"] = first_line
    
    return metadata


def chunk_documents(
    docs: List[Document],
    doc_id: str,
    strategy: str = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks with enhanced metadata
    
    Args:
        docs: List of documents to chunk
        doc_id: Unique identifier for the document
        strategy: Chunking strategy to use
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents with metadata
    """
    if strategy == ChunkingStrategy.RECURSIVE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == ChunkingStrategy.FIXED:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[""]
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    # Split documents
    chunks = splitter.split_documents(docs)
    
    # Enrich metadata for each chunk
    enriched_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk.metadata = _extract_metadata(chunk, doc_id, idx)
        enriched_chunks.append(chunk)
    
    logger.info(f"Created {len(enriched_chunks)} chunks from document {doc_id}")
    return enriched_chunks


def load_and_chunk_documents(
    filepaths: List[str],
    collection: str,
    strategy: str = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    skip_cached: bool = True
) -> tuple[List[Document], List[str]]:
    """
    Load documents from disk, chunk them, and prepare for ingestion
    
    Args:
        filepaths: List of file paths to process
        collection: Collection name for caching
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        skip_cached: Skip files that were already ingested
    
    Returns:
        Tuple of (chunked documents, list of processed filenames)
    """
    all_chunks = []
    processed_files = []
    
    for filepath in filepaths:
        filename = Path(filepath).name
        
        # Check cache
        if skip_cached and _is_file_cached(filepath, collection):
            logger.info(f"Skipping cached file: {filename}")
            continue
        
        try:
            # Load document
            docs = _load_document(filepath)
            
            # Generate document ID
            doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]
            
            # Chunk document
            chunks = chunk_documents(docs, doc_id, strategy, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            
            # Cache metadata
            _cache_file_metadata(filepath, collection, len(chunks))
            processed_files.append(filename)
            
            logger.info(f"Processed {filename}: {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            # Continue with other files
            continue
    
    logger.info(f"Total chunks prepared: {len(all_chunks)} from {len(processed_files)} files")
    return all_chunks, processed_files
# https://chat.deepseek.com/share/orjk4kh8sfa1tpxtqy
