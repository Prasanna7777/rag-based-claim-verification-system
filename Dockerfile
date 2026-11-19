FROM python:3.11-slim

WORKDIR /app

# Writable cache for the embedding model
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# CPU-only torch (the free host has no GPU), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model + embedding model downloaded at build time
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]