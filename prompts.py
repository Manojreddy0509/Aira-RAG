"""
Prompt Templates Module

Provides system and user prompt templates for RAG with context orchestration,
citation encouragement, and task-specific templates.
"""
from typing import List, Dict, Any
from langchain.schema import Document
from .vector_store import RetrievalResult


# PrivateGPT-style system prompt
SYSTEM_PROMPT_RAG = """You are a helpful AI assistant that answers questions based on provided document context.

Your responsibilities:
1. **Answer accurately**: Base your answers strictly on the provided context
2. **Cite sources**: When referencing information, mention the source document and page number
3. **Be thorough**: Provide detailed, well-reasoned answers
4. **Acknowledge uncertainty**: If the answer is not in the context, clearly state that you don't know
5. **Maintain context**: Consider all provided chunks together for a comprehensive answer

Remember: Your knowledge is limited to the provided context. Do not make assumptions or add information from outside sources."""


SYSTEM_PROMPT_CONCISE = """You are a helpful AI assistant that provides concise, accurate answers based on document context.

Answer the question using only the provided context. If the answer is not in the context, say "I don't have information about that in the provided documents." 

Keep answers brief but complete. Always cite the source document when referencing specific information."""


SYSTEM_PROMPT_DETAILED = """You are an expert AI assistant specializing in document analysis and question answering.

Guidelines:
- Provide comprehensive, well-structured answers
- Use bullet points or numbered lists for clarity when appropriate
- Quote relevant passages from the context when helpful
- Always cite sources with document name and page number
- If multiple sources support your answer, mention all of them
- If information is missing, explicitly state what is not covered in the documents
- For complex questions, break down your answer into logical sections

Your goal is to provide the most helpful and accurate response possible based solely on the provided context."""


def format_context_with_metadata(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into context string with metadata
    
    Args:
        results: List of retrieval results
    
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    context_parts = []
    for result in results:
        doc = result.document
        metadata = doc.metadata
        
        # Build metadata header
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        heading = metadata.get("heading", "")
        
        header = f"[Source: {source}, Page: {page}"
        if heading:
            header += f", Section: {heading}"
        header += f", Relevance: {result.score:.3f}]"
        
        # Format the chunk
        context_parts.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)


def format_context_simple(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into simple context string
    
    Args:
        results: List of retrieval results
    
    Returns:
        Simple formatted context string
    """
    return "\n\n".join([r.document.page_content for r in results])


def build_rag_prompt(
    question: str,
    results: List[RetrievalResult],
    system_prompt: str = SYSTEM_PROMPT_RAG,
    include_metadata: bool = True
) -> Dict[str, str]:
    """
    Build a complete RAG prompt with system and user messages
    
    Args:
        question: User's question
        results: Retrieved document chunks
        system_prompt: System prompt to use
        include_metadata: Whether to include metadata in context
    
    Returns:
        Dictionary with 'system' and 'user' messages
    """
    # Format context
    if include_metadata:
        context = format_context_with_metadata(results)
    else:
        context = format_context_simple(results)
    
    # Build user message
    user_message = f"""Based on the following context from documents, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return {
        "system": system_prompt,
        "user": user_message
    }


def build_chat_prompt(
    question: str,
    results: List[RetrievalResult],
    conversation_history: List[Dict[str, str]] = None,
    system_prompt: str = SYSTEM_PROMPT_RAG
) -> List[Dict[str, str]]:
    """
    Build a chat-style prompt with conversation history
    
    Args:
        question: Current question
        results: Retrieved document chunks
        conversation_history: Previous conversation turns
        system_prompt: System prompt to use
    
    Returns:
        List of message dictionaries for chat API
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current context and question
    context = format_context_with_metadata(results)
    user_message = f"""Based on the following context:

{context}

Question: {question}"""
    
    messages.append({"role": "user", "content": user_message})
    
    return messages


def build_summarization_prompt(documents: List[Document]) -> Dict[str, str]:
    """
    Build prompt for document summarization
    
    Args:
        documents: Documents to summarize
    
    Returns:
        Dictionary with system and user messages
    """
    system = """You are an expert at summarizing documents. Create a concise, comprehensive summary that captures the main points and key information."""
    
    content = "\n\n".join([doc.page_content for doc in documents])
    user = f"""Please summarize the following document content:

{content}

Provide a clear, structured summary with:
1. Main topic/purpose
2. Key points (3-5 bullet points)
3. Important details or conclusions"""
    
    return {"system": system, "user": user}


def extract_sources_from_results(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    """
    Extract source information from retrieval results
    
    Args:
        results: Retrieval results
    
    Returns:
        List of source dictionaries
    """
    sources = []
    seen = set()
    
    for result in results:
        metadata = result.document.metadata
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        
        # Create unique key to avoid duplicates
        key = f"{source}_{page}"
        if key not in seen:
            sources.append({
                "source": source,
                "page": page,
                "score": result.score,
                "heading": metadata.get("heading", "")
            })
            seen.add(key)
    
    return sources


def format_answer_with_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    """
    Format answer with source citations appended
    
    Args:
        answer: Generated answer
        sources: List of source dictionaries
    
    Returns:
        Answer with citations
    """
    if not sources:
        return answer
    
    # Build citations section
    citations = "\n\n**Sources:**\n"
    for idx, source in enumerate(sources, 1):
        source_name = source.get("source", "Unknown")
        page = source.get("page", "N/A")
        citations += f"{idx}. {source_name} (Page {page})\n"
    
    return answer + citations
