"""
VectorStore Interface Layer

Provides unified interface for vector storage and retrieval with scoring,
metadata filtering, and optional reranking.
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to meet ChromaDB requirements:
    1. 3-63 characters
    2. Starts and ends with alphanumeric
    3. Only alphanumeric, underscores, or hyphens
    4. No consecutive periods
    5. Not a valid IPv4 address
    """
    if not name:
        return "collection_default"
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Remove consecutive underscores/hyphens
    sanitized = re.sub(r'[_-]{2,}', '_', sanitized)
    
    # Ensure starts and ends with alphanumeric
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    
    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = f"collection_{sanitized}"
    
    # Ensure maximum length
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Re-ensure ends with alphanumeric after truncation
        sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    
    # Fallback if somehow still invalid
    if not sanitized or len(sanitized) < 3:
        sanitized = "collection_default"
    
    return sanitized

VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
VECTOR_DIR = os.path.abspath(VECTOR_DIR)

_embeddings_cache = None


@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "score": self.score,
            "rank": self.rank
        }


class VectorStore:
    """
    Unified VectorStore interface for document ingestion and retrieval
    
    Supports:
    - Document ingestion with batching
    - Similarity search with configurable top-k
    - Score normalization and filtering
    - Optional reranking
    """
    
    def __init__(self, collection_name: str, embeddings: Optional[HuggingFaceEmbeddings] = None):
        """
        Initialize VectorStore
        
        Args:
            collection_name: Name of the collection
            embeddings: Embedding model (uses default if None)
        """
        # Sanitize collection name for ChromaDB
        self.collection_name = sanitize_collection_name(collection_name)
        if self.collection_name != collection_name:
            logger.info(f"Sanitized collection name: '{collection_name}' -> '{self.collection_name}'")
        
        self.embeddings = embeddings or get_embeddings()
        self._store = self._initialize_store()
        logger.info(f"Initialized VectorStore for collection: {self.collection_name}")
    
    def _initialize_store(self) -> Chroma:
        """Initialize the underlying Chroma store"""
        os.makedirs(VECTOR_DIR, exist_ok=True)
        return Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=VECTOR_DIR
        )
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for ingestion
        
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        try:
            # Add in batches to handle large document sets
            total_added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self._store.add_documents(batch)
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Persist to disk
            self._store.persist()
            logger.info(f"Successfully added {total_added} documents to {self.collection_name}")
            return total_added
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            metadata_filter: Filter by metadata fields
        
        Returns:
            List of retrieval results with scores
        """
        try:
            # Perform similarity search with scores
            search_kwargs = {"k": top_k}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
            
            # Use similarity_search_with_score for scoring
            results = self._store.similarity_search_with_score(query, k=top_k)
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for rank, (doc, distance) in enumerate(results):
                # Convert distance to similarity score (Chroma uses L2 distance)
                # Lower distance = higher similarity
                # Normalize to 0-1 range (approximate)
                score = 1.0 / (1.0 + distance)
                
                # Apply score threshold filter
                if score_threshold and score < score_threshold:
                    continue
                
                # Store score in metadata for later use
                doc.metadata["score"] = score
                doc.metadata["distance"] = distance
                
                retrieval_results.append(
                    RetrievalResult(document=doc, score=score, rank=rank + 1)
                )
            
            logger.info(f"Retrieved {len(retrieval_results)} documents for query (top_k={top_k})")
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 4,
        rerank_top_n: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with two-stage retrieval and reranking
        
        First retrieves more candidates (rerank_top_n), then reranks and returns top_k
        
        Args:
            query: Query string
            top_k: Final number of documents to return
            rerank_top_n: Number of candidates to retrieve before reranking
            score_threshold: Minimum similarity score
        
        Returns:
            List of reranked retrieval results
        """
        # Retrieve more candidates
        candidates = self.retrieve(
            query=query,
            top_k=rerank_top_n,
            score_threshold=score_threshold
        )
        
        if not candidates:
            return []
        
        # Simple reranking: boost documents with query terms in heading or title
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for result in candidates:
            boost = 0.0
            
            # Boost if query terms appear in heading
            heading = result.document.metadata.get("heading", "").lower()
            if heading and any(term in heading for term in query_terms):
                boost += 0.1
            
            # Boost if query terms appear in page content
            content_lower = result.document.page_content.lower()
            matching_terms = sum(1 for term in query_terms if term in content_lower)
            boost += 0.02 * matching_terms
            
            # Apply boost to score
            result.score = min(1.0, result.score + boost)
        
        # Re-sort by boosted score and take top_k
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Update ranks
        for idx, result in enumerate(reranked):
            result.rank = idx + 1
        
        logger.info(f"Reranked {len(candidates)} candidates to {len(reranked)} results")
        return reranked
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self._store.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get cached embeddings model"""
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info("Initializing embeddings model: sentence-transformers/all-MiniLM-L6-v2")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_cache


def ensure_vector_directory():
    """Ensure vector storage directory exists"""
    os.makedirs(VECTOR_DIR, exist_ok=True)
