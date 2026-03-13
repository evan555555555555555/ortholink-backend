FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for WeasyPrint + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libcairo2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.1.3 && poetry config virtualenvs.create false

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy app code
COPY . .

# Download FAISS vector database from GitHub Release (33,949 chunks, 18 markets)
RUN mkdir -p /app/data/embeddings && \
    curl -L -o /app/data/embeddings/faiss.index \
      https://github.com/evan555555555555555/ortholink-backend/releases/download/v1.0.0/faiss.index && \
    curl -L -o /app/data/embeddings/metadata.json \
      https://github.com/evan555555555555555/ortholink-backend/releases/download/v1.0.0/metadata.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
