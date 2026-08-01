# Single-stage image: FastAPI backend + Postgres-queue worker.
#
# BACKEND_DIR (app/config.py) resolves to Path(__file__).resolve().parent.parent,
# i.e. the parent of the installed `app` package. `pip install -e .` leaves the
# package in place at /app/app (an editable install just points back at the
# source tree, it doesn't copy files into site-packages) -- so BACKEND_DIR still
# resolves to /app and PROMPTS_DIR (extraction.py) still finds /app/prompts.
FROM python:3.12-slim

WORKDIR /app

# Dependency-relevant files first so the pip install layer only busts on
# code/dependency changes, not on prompt or script edits below.
COPY pyproject.toml ./
COPY app ./app

# Runtime deps only -- [dev] (pytest/ruff/pillow) never ships in the image.
RUN pip install --no-cache-dir -e .

COPY alembic.ini ./
COPY alembic ./alembic
COPY prompts ./prompts
COPY scripts ./scripts

# Non-root user. /data/uploads is created and chowned before switching users
# so the named volume Docker mounts over it inherits writable ownership.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/uploads \
    && chown -R appuser:appuser /data

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
