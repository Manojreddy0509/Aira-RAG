# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for torch and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The build context is the simple_rag/ directory
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the entire app directory into a package named simple_rag
COPY . /app/simple_rag

ENV OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3.1 \
    PREFER_OPENAI=false

EXPOSE 8000
CMD ["uvicorn", "simple_rag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
