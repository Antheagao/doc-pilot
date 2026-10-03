variable "project_id" {
  description = "The GCP project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Cloud SQL, the image registry and the uploads bucket."
  type        = string
  default     = "us-central1"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "doc-pilot"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name))
    error_message = "name must be 2-21 characters: lowercase letters, digits and hyphens, starting with a letter."
  }
}

# ---- what to run ----------------------------------------------------------------

variable "backend_image_tag" {
  description = <<-EOT
    Tag of the backend image in this deployment's Artifact Registry
    repository (api, worker and migrations all run it). Null on the first
    apply, which creates the registry, database and bucket but no
    services -- push the images, then apply again with the tags set.
  EOT
  type        = string
  default     = null
}

variable "frontend_image_tag" {
  description = "Tag of the frontend image; null deploys no frontend. Build it with NEXT_PUBLIC_API_URL set to the api_url output."
  type        = string
  default     = null
}

# ---- exposure ---------------------------------------------------------------------

variable "public" {
  description = <<-EOT
    Serve the frontend and API to anyone on the internet. doc-pilot has no
    user accounts (see the README's Security model): a public deployment
    lets anyone upload documents and spend the model budget, bounded only
    by daily_budget_usd and the rate limit. Off by default: only callers
    with roles/run.invoker reach the services, e.g. through
    `gcloud run services proxy` (see infra/terraform/README.md).
  EOT
  type        = bool
  default     = false
}

# ---- spend guards (app/budget.py, app/ratelimit.py, app/evals/online.py) ----------

variable "daily_budget_usd" {
  description = "Uploads and questions get 429 once the day's (UTC) recorded model spend reaches this; 0 turns the cap off."
  type        = number
  default     = 5

  validation {
    condition     = var.daily_budget_usd >= 0
    error_message = "daily_budget_usd can't be negative."
  }
}

variable "ask_rate_limit_per_minute" {
  description = "Requests per minute per client address on /ask and the document chat (0 = off)."
  type        = number
  default     = 10
}

variable "ask_judge_sample_rate" {
  description = "Share (0-1) of answered questions graded in the background for groundedness -- one more model call each."
  type        = number
  default     = 0

  validation {
    condition     = var.ask_judge_sample_rate >= 0 && var.ask_judge_sample_rate <= 1
    error_message = "ask_judge_sample_rate is a probability, 0-1."
  }
}

# ---- sizing -------------------------------------------------------------------------

variable "api_max_instances" {
  description = <<-EOT
    Upper bound on API instances. The rate limiter is in memory, per
    instance, so N instances allow up to N times the per-minute limit.
  EOT
  type        = number
  default     = 2
}

variable "db_tier" {
  description = "Cloud SQL machine tier. db-f1-micro (shared core) suits a demo; use a db-custom-* tier for real load."
  type        = string
  default     = "db-f1-micro"
}

variable "db_deletion_protection" {
  description = "Refuse to destroy the database (and its data) on `terraform destroy`."
  type        = bool
  default     = true
}

variable "db_password_version" {
  description = <<-EOT
    The database password is generated during apply and written only to
    Cloud SQL and Secret Manager -- never to Terraform state. Bump this to
    rotate it, then redeploy the services so they read the new secret.
  EOT
  type        = number
  default     = 1
}

variable "worker_reclaim_after_seconds" {
  description = <<-EOT
    A Cloud Run rollout can briefly run the old and the new worker
    together, so the worker only reclaims jobs that have been processing
    longer than this (app/worker.py). Longer than any job takes.
  EOT
  type        = number
  default     = 1800
}
