# Build stage
FROM python:3.12-slim as builder

WORKDIR /app
COPY requirements.txt .

# Utilisation du cache pour accélérer les installations (BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# OpenCV runtime dependencies (libgl1 replaces libgl1-mesa-glx in newer Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code + modèle de production (depuis models/)
COPY src/ /app/src/
COPY models/ai-sharp-exp-prod.pt /app/models/ai-sharp-exp-prod.pt

# Configuration
ENV MODEL_PATH=/app/models/ai-sharp-exp-prod.pt
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
