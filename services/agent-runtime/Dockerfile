# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=info

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r stratoshub && useradd -r -g stratoshub -s /bin/bash stratoshub

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=stratoshub:stratoshub src/ ./src/
COPY --chown=stratoshub:stratoshub scripts/ ./scripts/
COPY --chown=stratoshub:stratoshub config/ ./config/
COPY --chown=stratoshub:stratoshub pyproject.toml .
COPY --chown=stratoshub:stratoshub README.md .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data /app/temp && \
    chown -R stratoshub:stratoshub /app

# Switch to non-root user
USER stratoshub

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "src.main:app", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "4", "--max-requests", "1000", "--max-requests-jitter", "50", "--timeout", "300", "--keep-alive", "2", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]

# Labels for metadata
LABEL maintainer="StratosHub Team" \
      version="0.1.0" \
      description="StratosHub AI Agent Runtime Service" \
      org.opencontainers.image.source="https://github.com/stratoshub/stratoshub" \
      org.opencontainers.image.documentation="https://docs.stratoshub.io" \
      org.opencontainers.image.licenses="MIT" 