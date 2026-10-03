# Everything doc-pilot needs before its containers run: APIs, an image
# registry, the Postgres database (pgvector is a Cloud SQL extension the
# first migration enables), a bucket for uploads, the secrets, and the
# service accounts. The containers themselves are in services.tf.

data "google_project" "this" {}

locals {
  services = [
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
  ]

  registry = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"

  # Cloud Run's deterministic URLs (https://SERVICE-PROJECT_NUMBER.REGION.run.app),
  # computed rather than read from the services so the API can allow the
  # frontend's origin and the frontend can be built against the API's URL
  # without a dependency cycle.
  api_name = "${var.name}-api"
  web_name = "${var.name}-web"
  api_url  = "https://${local.api_name}-${data.google_project.this.number}.${var.region}.run.app"
  web_url  = "https://${local.web_name}-${data.google_project.this.number}.${var.region}.run.app"
}

resource "google_project_service" "apis" {
  for_each = toset(local.services)

  service            = each.value
  disable_on_destroy = false
}

# ---- images ---------------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  repository_id = var.name
  location      = var.region
  format        = "DOCKER"
  description   = "doc-pilot backend and frontend images"

  depends_on = [google_project_service.apis]
}

# ---- database -------------------------------------------------------------------

resource "google_sql_database_instance" "db" {
  name                = "${var.name}-db"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = var.db_deletion_protection

  settings {
    tier              = var.db_tier
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    # A public IP with no authorized networks: nothing can connect over
    # it directly. Cloud Run reaches the instance through its built-in
    # Cloud SQL connector, which authorizes the caller by IAM
    # (roles/cloudsql.client) and encrypts the connection -- no VPC needed.
    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "docpilot" {
  name     = "docpilot"
  instance = google_sql_database_instance.db.name
}

# Ephemeral: generated during apply and sent to Cloud SQL and Secret
# Manager through write-only arguments, so the password is never stored in
# Terraform state. Rotate it with db_password_version.
ephemeral "random_password" "db" {
  length  = 32
  special = false # it goes inside a URL
}

resource "google_sql_user" "docpilot" {
  name                = "docpilot"
  instance            = google_sql_database_instance.db.name
  password_wo         = ephemeral.random_password.db.result
  password_wo_version = var.db_password_version
}

# ---- uploads ----------------------------------------------------------------------

# The API writes each upload to UPLOAD_DIR and the worker reads it back from
# the same path; on Cloud Run both mount this bucket there (Cloud Storage
# FUSE), in place of compose's shared volume.
resource "google_storage_bucket" "uploads" {
  name                        = "${var.project_id}-${var.name}-uploads"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  depends_on = [google_project_service.apis]
}

# ---- secrets ------------------------------------------------------------------------

resource "google_secret_manager_secret" "database_url" {
  secret_id = "${var.name}-database-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# asyncpg reaches Cloud SQL through the Unix socket Cloud Run mounts at
# /cloudsql/<connection name>.
resource "google_secret_manager_secret_version" "database_url" {
  secret                 = google_secret_manager_secret.database_url.id
  secret_data_wo         = "postgresql+asyncpg://${google_sql_user.docpilot.name}:${ephemeral.random_password.db.result}@/${google_sql_database.docpilot.name}?host=/cloudsql/${google_sql_database_instance.db.connection_name}"
  secret_data_wo_version = var.db_password_version
}

# The key itself is added outside Terraform, so it is never in a plan or
# state file:
#   printf %s "$ANTHROPIC_API_KEY" | gcloud secrets versions add doc-pilot-anthropic-api-key --data-file=-
resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "${var.name}-anthropic-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# ---- identities -----------------------------------------------------------------------

# The API, the worker and the migration job: the database, both secrets
# and the uploads bucket -- and nothing else in the project.
resource "google_service_account" "backend" {
  account_id   = "${var.name}-backend"
  display_name = "doc-pilot API, worker and migrations"
}

resource "google_project_iam_member" "backend_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = google_service_account.backend.member
}

resource "google_secret_manager_secret_iam_member" "backend" {
  for_each = {
    database_url      = google_secret_manager_secret.database_url.id
    anthropic_api_key = google_secret_manager_secret.anthropic_api_key.id
  }

  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.backend.member
}

resource "google_storage_bucket_iam_member" "backend_uploads" {
  bucket = google_storage_bucket.uploads.name
  role   = "roles/storage.objectUser"
  member = google_service_account.backend.member
}

# The frontend only serves static pages and the browser calls the API
# directly, so its identity needs no roles at all.
resource "google_service_account" "frontend" {
  account_id   = "${var.name}-web"
  display_name = "doc-pilot frontend"
}
