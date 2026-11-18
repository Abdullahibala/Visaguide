# VisaGuide API image.
#
# The search index is built and both models are downloaded DURING THE BUILD, so the
# container starts with everything it needs: no first-request delay, and no runtime
# dependency on Hugging Face (HF_HUB_OFFLINE=1). The image contains public data and
# code only; the API key arrives at runtime as an environment variable / secret.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app

# 1. Dependencies first (this layer is cached until pyproject.toml changes).
#    CPU-only torch from PyTorch's index: the default wheel bundles CUDA (~2.5 GB).
COPY pyproject.toml ./
COPY src/visaguide/__init__.py src/visaguide/__init__.py
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system -e .

# 2. Code, config and the processed corpus (raw downloads are not needed).
COPY src ./src
COPY configs ./configs
COPY scripts/build_index.py ./scripts/build_index.py
COPY data/processed ./data/processed
COPY data/manifest.csv ./data/manifest.csv

# 3. Build the index for the serving configuration and cache the reranker model.
RUN python scripts/build_index.py --strategy structure \
 && python -c "from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 4. Run as an unprivileged user; logs/ must be writable by it.
RUN useradd --create-home --uid 10001 app && mkdir -p logs && chown -R app:app /app
USER app
ENV HF_HUB_OFFLINE=1 \
    PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# Cloud Run sets $PORT. One worker: the models are loaded per process, and the
# rate limiter / budget state lives in this process's memory.
CMD ["sh", "-c", "uvicorn visaguide.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
