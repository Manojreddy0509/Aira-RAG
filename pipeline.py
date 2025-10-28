"""
Query Pipeline Module

Provides end-to-end RAG pipeline that chains:
1. Document retrieval
2. Prompt orchestration
3. LLM calling
4. Response formatting
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .vector_store import VectorStore, RetrievalResult
from .prompts import (
    build_rag_prompt,
    extract_sources_from_results,
    format_answer_with_citations,
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_DETAILED
)
from .llm import call_llm_with_context, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Complete response from query pipeline"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    latency_breakdown: Dict[str, float]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_results": self.retrieval_results,
            "model_info": self.model_info,
            "latency_breakdown": self.latency_breakdown,
            "error": self.error
        }


class RAGPipeline:
    """
    End-to-end RAG pipeline
    
    Handles the complete flow from query to answer:
    - Document retrieval with optional reranking
    - Context preparation with metadata
    - Prompt building with templates
    - LLM invocation with fallback
    - Response formatting with citations
    """
    
    def __init__(
        self,
        collection_name: str,
        system_prompt: str = SYSTEM_PROMPT_RAG,
        use_reranking: bool = False,
        temperature: float = 0.2
    ):
        """
        Initialize RAG pipeline
        
        Args:
            collection_name: Vector store collection name
            system_prompt: System prompt template to use
            use_reranking: Whether to use reranking
            temperature: LLM temperature
        """
        self.collection_name = collection_name
        self.system_prompt = system_prompt
        self.use_reranking = use_reranking
        self.temperature = temperature
        self.vector_store = VectorStore(collection_name)
        logger.info(f"Initialized RAG pipeline for collection: {collection_name}")
    
    def query(
        self,
        question: str,
        top_k: int = 4,
        max_tokens: Optional[int] = None,
        include_citations: bool = True
    ) -> QueryResponse:
        """
        Execute complete RAG query pipeline
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens in response
            include_citations: Whether to append source citations
        
        Returns:
            QueryResponse with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        latency_breakdown = {}
        start_total = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            start_retrieval = time.time()
            if self.use_reranking:
                retrieval_results = self.vector_store.retrieve_with_rerank(
                    query=question,
                    top_k=top_k,
                    rerank_top_n=min(top_k * 3, 20)
                )
            else:
                retrieval_results = self.vector_store.retrieve(
                    query=question,
                    top_k=top_k
                )
            latency_breakdown["retrieval"] = time.time() - start_retrieval
            
            if not retrieval_results:
                logger.warning("No documents retrieved")
                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents.",
                    sources=[],
                    retrieval_results=[],
                    model_info={"model": "none", "provider": "none"},
                    latency_breakdown=latency_breakdown,
                    error="No documents retrieved"
                )
            
            logger.info(f"Retrieved {len(retrieval_results)} documents in {latency_breakdown['retrieval']:.2f}s")
            
            # Step 2: Build prompt with context
            start_prompt = time.time()
            prompt_dict = build_rag_prompt(
                question=question,
                results=retrieval_results,
                system_prompt=self.system_prompt,
                include_metadata=True
            )
            latency_breakdown["prompt_building"] = time.time() - start_prompt
            
            # Step 3: Call LLM
            start_llm = time.time()
            llm_response = call_llm_with_context(
                system_prompt=prompt_dict["system"],
                user_prompt=prompt_dict["user"],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            latency_breakdown["llm_call"] = llm_response.latency_seconds
            
            if llm_response.error:
                logger.error(f"LLM call failed: {llm_response.error}")
                return QueryResponse(
                    answer="",
                    sources=[],
                    retrieval_results=[r.to_dict() for r in retrieval_results],
                    model_info={"model": llm_response.model, "provider": llm_response.provider},
                    latency_breakdown=latency_breakdown,
                    error=llm_response.error
                )
            
            logger.info(f"LLM response generated in {llm_response.latency_seconds:.2f}s")
            
            # Step 4: Format response
            start_format = time.time()
            sources = extract_sources_from_results(retrieval_results)
            answer = llm_response.content
            
            if include_citations:
                answer = format_answer_with_citations(answer, sources)
            
            latency_breakdown["formatting"] = time.time() - start_format
            latency_breakdown["total"] = time.time() - start_total
            
            logger.info(f"Query completed in {latency_breakdown['total']:.2f}s")
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                retrieval_results=[r.to_dict() for r in retrieval_results],
                model_info={
                    "model": llm_response.model,
                    "provider": llm_response.provider,
                    "temperature": self.temperature
                },
                latency_breakdown=latency_breakdown
            )
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            latency_breakdown["total"] = time.time() - start_total
            return QueryResponse(
                answer="",
                sources=[],
                retrieval_results=[],
                model_info={"model": "unknown", "provider": "unknown"},
                latency_breakdown=latency_breakdown,
                error=str(e)
            )


def query_document(
    collection: str,
    question: str,
    top_k: int = 4,
    temperature: float = 0.2,
    use_reranking: bool = False,
    system_prompt: str = SYSTEM_PROMPT_RAG
) -> Dict[str, Any]:
    """
    Simplified function for querying documents
    
    Args:
        collection: Collection name
        question: User question
        top_k: Number of documents to retrieve
        temperature: LLM temperature
        use_reranking: Whether to use reranking
        system_prompt: System prompt template
    
    Returns:
        Dictionary with answer and metadata
    """
    pipeline = RAGPipeline(
        collection_name=collection,
        system_prompt=system_prompt,
        use_reranking=use_reranking,
        temperature=temperature
    )
    
    response = pipeline.query(question, top_k=top_k)
    return response.to_dict()


def ingest_documents(
    collection: str,
    file_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    skip_cached: bool = True
) -> Dict[str, Any]:
    """
    Ingest documents into vector store
    
    Args:
        collection: Collection name
        file_paths: List of file paths to ingest
        chunk_size: Chunk size for splitting
        chunk_overlap: Overlap between chunks
        skip_cached: Skip already ingested files
    
    Returns:
        Dictionary with ingestion results
    """
    from .ingest import load_and_chunk_documents
    
    logger.info(f"Starting ingestion for collection: {collection}")
    start_time = time.time()
    
    try:
        # Load and chunk documents
        chunks, processed_files = load_and_chunk_documents(
            filepaths=file_paths,
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_cached=skip_cached
        )
        
        if not chunks:
            logger.warning("No new documents to ingest")
            return {
                "collection": collection,
                "chunks": 0,
                "files": [],
                "latency": time.time() - start_time
            }
        
        # Add to vector store
        vector_store = VectorStore(collection)
        num_added = vector_store.add_documents(chunks)
        
        latency = time.time() - start_time
        logger.info(f"Ingestion completed: {num_added} chunks from {len(processed_files)} files in {latency:.2f}s")
        
        return {
            "collection": collection,
            "chunks": num_added,
            "files": processed_files,
            "latency": latency
        }
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return {
            "collection": collection,
            "chunks": 0,
            "files": [],
            "latency": time.time() - start_time,
            "error": str(e)
        }
