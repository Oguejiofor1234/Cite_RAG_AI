# ==============================================================================
# Cite_RAG_AI — Multi-Stage Production Dockerfile
# Provides high-performance PyMuPDF rendering, Tesseract OCR, FAISS vector index,
# and secure non-root Streamlit execution.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools and header packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for clean dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

LABEL maintainer="Ogueji" \
      description="Cite_RAG_AI — Agentic Document Intelligence & Precision Evidence Retrieval" \
      version="1.0.0"

# Install system runtime libraries:
# - tesseract-ocr, tesseract-ocr-eng: scanned PDF optical character recognition fallback
# - libgl1, libglib2.0-0: PyMuPDF and Pillow headless graphics support
# - curl, ca-certificates: healthcheck probe and secure HTTPS transport
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application directories and files
COPY --chown=appuser:appgroup app_multi.py .
COPY --chown=appuser:appgroup app_single.py .
COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup rag/ ./rag/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup .streamlit/ ./.streamlit/

# Create runtime directories for data uploads and vector storage
RUN mkdir -p /app/data /app/data/uploads /app/data/indexes && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Configure environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck monitoring Streamlit core health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_multi.py"]
