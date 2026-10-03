# The backend image as Cloud Run runs it: backend/Dockerfile's image plus
# the committed eval-run artifacts that GET /stats and the monitoring
# dashboard read (docker compose mounts them instead). The backend image
# alone can't hold them -- its build context is backend/. From the repo
# root, after building backend/:
#
#   docker build -f infra/backend.Dockerfile \
#     --build-arg BACKEND_IMAGE=doc-pilot-backend:build -t REGISTRY/backend:TAG .
#
# The repo root's .dockerignore sends only evals/results.
ARG BACKEND_IMAGE
FROM ${BACKEND_IMAGE}

# app/evals resolves EVALS_DIR to /evals, beside the /app install.
COPY evals/results /evals/results
