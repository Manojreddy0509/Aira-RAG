import os
import time
import logging
import requests
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PREFER_OPENAI = os.environ.get("PREFER_OPENAI", "false").lower() in {"1", "true", "yes"}

# Legacy system prompt (kept for backward compatibility)
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about provided documents. "
    "Use the given context to answer succinctly. If the answer is not in the context, say you don't know."
)


@dataclass
class LLMResponse:
    """Response from LLM call"""
    content: str
    model: str
    provider: str  # 'ollama' or 'openai'
    latency_seconds: float
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


def _call_ollama_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    stream: bool = False
) -> str:
    """Call Ollama chat endpoint"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Ollama chat API error: {e}")
        raise


def _call_ollama_generate(prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
    """Call Ollama generate endpoint (fallback)"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        logger.error(f"Ollama generate API error: {e}")
        raise


def _stream_ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> Generator[str, None, None]:
    """Stream responses from Ollama chat endpoint"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=180, stream=True)
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        raise


def _call_openai_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False
) -> str:
    """Call OpenAI chat API"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def _stream_openai_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    model: Optional[str] = None
) -> Generator[str, None, None]:
    """Stream responses from OpenAI chat API"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180, stream=True)
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    if line_str.strip() == 'data: [DONE]':
                        break
                    import json
                    data = json.loads(line_str[6:])
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        raise


def call_llm_with_context(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    stream: bool = False
) -> LLMResponse:
    """
    Unified LLM call with context orchestration
    
    Args:
        system_prompt: System message
        user_prompt: User message
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        model: Specific model to use (optional)
        stream: Whether to stream response
    
    Returns:
        LLMResponse with answer and metadata
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    start_time = time.time()
    
    # Try OpenAI first if preferred
    if PREFER_OPENAI and OPENAI_API_KEY:
        try:
            logger.info(f"Calling OpenAI API (model: {model or 'default'})")
            content = _call_openai_chat(messages, temperature, model, max_tokens, stream)
            latency = time.time() - start_time
            logger.info(f"OpenAI API call completed in {latency:.2f}s")
            return LLMResponse(
                content=content,
                model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                provider="openai",
                latency_seconds=latency
            )
        except Exception as e:
            logger.warning(f"OpenAI API failed, falling back to Ollama: {e}")
    
    # Fallback to Ollama
    try:
        logger.info(f"Calling Ollama API (model: {OLLAMA_MODEL})")
        
        # Try chat endpoint first
        try:
            content = _call_ollama_chat(messages, temperature, max_tokens, stream)
            latency = time.time() - start_time
            logger.info(f"Ollama chat API call completed in {latency:.2f}s")
            return LLMResponse(
                content=content,
                model=OLLAMA_MODEL,
                provider="ollama",
                latency_seconds=latency
            )
        except Exception as chat_error:
            logger.warning(f"Ollama chat endpoint failed, trying generate: {chat_error}")
            # Fallback to generate endpoint
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            content = _call_ollama_generate(combined_prompt, temperature, max_tokens)
            latency = time.time() - start_time
            logger.info(f"Ollama generate API call completed in {latency:.2f}s")
            return LLMResponse(
                content=content,
                model=OLLAMA_MODEL,
                provider="ollama",
                latency_seconds=latency
            )
    
    except Exception as e:
        logger.error(f"All LLM API calls failed: {e}")
        latency = time.time() - start_time
        return LLMResponse(
            content="",
            model="unknown",
            provider="error",
            latency_seconds=latency,
            error=str(e)
        )


def generate_answer(question: str, context: str, temperature: float = 0.2) -> str:
    """
    Legacy function for backward compatibility
    
    Args:
        question: User question
        context: Retrieved context
        temperature: Temperature parameter
    
    Returns:
        Generated answer string
    """
    system = SYSTEM_PROMPT
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    
    response = call_llm_with_context(system, user, temperature)
    return response.content if response.content else "Error generating answer."
