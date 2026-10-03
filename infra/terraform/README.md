# Deploying doc-pilot to Google Cloud (Terraform)

This configuration runs doc-pilot on Cloud Run with Cloud SQL. The pieces are the same as `docker compose up`; only where each runs changes.

| docker compose | Here |
| --- | --- |
| `db` (Postgres 16 + pgvector) | Cloud SQL for PostgreSQL 16. The first migration runs `CREATE EXTENSION vector`, which Cloud SQL supports. |
| `migrate` (one-shot) | Cloud Run job `doc-pilot-migrate`. Terraform runs it on every new backend image and waits for it to succeed before rolling out the API and worker. |
| `api` | Cloud Run service `doc-pilot-api`. |
| `worker` (exactly one) | Cloud Run worker pool `doc-pilot-worker` with one instance. A worker pool pulls work and serves no requests. |
| `frontend` | Cloud Run service `doc-pilot-web`. |
| `uploads` volume shared by api + worker | A Cloud Storage bucket that both mount at `/data/uploads` (Cloud Storage FUSE). |
| `models` volume | Each container's in-memory `/tmp`. The pinned embedding model (77 MB, checksummed) downloads on cold start. |
| `.env` | Secret Manager: `DATABASE_URL`, `ANTHROPIC_API_KEY`. |

**State.** The database password is generated during `apply` and passed to Cloud SQL and Secret Manager through write-only arguments. It is never written to Terraform state, and neither is the Anthropic key, which you add with `gcloud`. State is local by default. For anything shared, put it in a bucket by adding a `backend "gcs" { bucket = "..." }` block to `versions.tf`.

## Before you start

You need:

- a GCP project with billing enabled;
- `gcloud` authenticated (`gcloud auth login` and `gcloud auth application-default login`);
- Terraform 1.11 or later, for write-only arguments and ephemeral resources;
- Docker.

The steps below use `us-central1`. If you use another region, swap it everywhere.

## 1. The foundation

```sh
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set project_id
terraform init
terraform apply
```

With no image tags set, this creates:

- the APIs;
- the image registry;
- Cloud SQL (this takes about 10 minutes);
- the uploads bucket;
- the secrets and service accounts.

It doesn't create any containers, because Cloud Run rejects an image that hasn't been pushed yet.

## 2. The API key

```sh
printf %s "$ANTHROPIC_API_KEY" | gcloud secrets versions add "$(terraform output -raw anthropic_api_key_secret)" --data-file=-
```

## 3. The images

Build them for `linux/amd64`, which is what Cloud Run runs. This matters on Apple silicon. Run these commands from the repo root:

```sh
REGISTRY=$(terraform -chdir=infra/terraform output -raw registry)
gcloud auth configure-docker us-central1-docker.pkg.dev

# Backend: the image itself, then the committed eval results on top
# (for /stats and the monitoring dashboard; see infra/backend.Dockerfile).
docker build --platform linux/amd64 -t doc-pilot-backend:build backend
docker build --platform linux/amd64 -f infra/backend.Dockerfile \
  --build-arg BACKEND_IMAGE=doc-pilot-backend:build -t "$REGISTRY/backend:v1" .
docker push "$REGISTRY/backend:v1"

# Frontend: the API's URL is compiled into the bundle (NEXT_PUBLIC_API_URL).
#   public deployment:  the api_url output
#   private deployment: http://localhost:8000, the local proxy below
docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL="$(terraform -chdir=infra/terraform output -raw api_url)" \
  -t "$REGISTRY/frontend:v1" frontend
docker push "$REGISTRY/frontend:v1"
```

## 4. Deploy

Set `backend_image_tag = "v1"` and `frontend_image_tag = "v1"` in `terraform.tfvars`, then run:

```sh
terraform apply
```

This runs the migrations and waits for them to finish. Then it deploys the API, the worker and the frontend. To ship a new version, push new tags and apply again.

## Reaching it

**Private** (`public = false`, the default). Only principals with `roles/run.invoker` on the services can call them. Each service is reachable at its URL with an identity token, or through a local authenticated proxy:

```sh
gcloud run services proxy doc-pilot-api --region us-central1 --port 8000 &
gcloud run services proxy doc-pilot-web --region us-central1 --port 3000
# open http://localhost:3000 (built with NEXT_PUBLIC_API_URL=http://localhost:8000)
```

**Public** (`public = true`). Both services skip the IAM check, so read the root README's *Security model* first. doc-pilot has no user accounts, so anyone with the URL can upload documents and ask questions, and both of those are billed. What bounds the spend:

- **`daily_budget_usd`**: a hard daily cap.
- **The per-client rate limit** (`ask_rate_limit_per_minute`): every request reaches the container through Cloud Run's front end, so the address the API sees isn't the browser's. The limit therefore behaves more like one shared limit than a per-client one. The API deliberately doesn't trust `X-Forwarded-For`, because a client can set that header to any key it likes. Edge rate limiting (Cloud Armor in front of a load balancer) is the fix if that matters.
- **Per-question step and dollar caps.**

## How it fits together

- **Database.** Cloud SQL has a public IP but no authorized networks, so nothing connects to it directly. Cloud Run's built-in Cloud SQL connector authorizes callers by IAM (`roles/cloudsql.client`) and encrypts the connection. The app connects through the Unix socket the connector mounts at `/cloudsql/...`. No VPC is needed.
- **Identities.**
  - The API, worker and migration job share one service account. It can use the database, read the two secrets, and read and write the uploads bucket, and nothing else.
  - The frontend's service account has no roles. The browser calls the API directly.
- **Worker rollouts.** A rollout can briefly run the old worker alongside the new one, and `app/worker.py` assumes a single worker. To cover that, the worker here:
  - reclaims a stuck job only after it has been processing for `worker_reclaim_after_seconds` (30 minutes), checking every minute, so it never takes over a job that is still being worked on;
  - stops claiming work on SIGTERM.
- **Rate limits per instance.** The rate limiter keeps its state in memory, so `api_max_instances` multiplies the per-minute limit.
- **Tracing** is off. Set `OTEL_EXPORTER_OTLP_ENDPOINT` on the services to export to a collector, for example one forwarding to Cloud Trace.

## Tests

`terraform test` runs `tests/deploy.tftest.hcl` against mocked providers, so it needs no credentials and creates nothing. The tests check that:

- nothing deploys without an image;
- the API and worker run the pushed image;
- there is exactly one worker;
- a new image triggers a migration;
- the private and public CORS and IAM settings are right;
- credentials come only from Secret Manager.

CI runs `fmt`, `validate` and `test` on every push.

## Tearing down

```sh
terraform apply -var db_deletion_protection=false   # it is on by default
terraform destroy
```

The uploads bucket refuses to be destroyed while it holds objects (`force_destroy = false`). Empty it first if you mean it.
