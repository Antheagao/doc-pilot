# The containers: a migration job, the API, the queue worker and the
# frontend. None exist until backend_image_tag is set (see variables.tf),
# because Cloud Run refuses an image that hasn't been pushed yet.

locals {
  deploy_backend  = var.backend_image_tag != null
  deploy_frontend = var.frontend_image_tag != null && local.deploy_backend

  backend_image  = "${local.registry}/backend:${coalesce(var.backend_image_tag, "unset")}"
  frontend_image = "${local.registry}/frontend:${coalesce(var.frontend_image_tag, "unset")}"

  uploads_dir = "/data/uploads"

  # Settings every backend container shares (app/config.py).
  backend_env = {
    UPLOAD_DIR                = local.uploads_dir
    DAILY_BUDGET_USD          = tostring(var.daily_budget_usd)
    ASK_RATE_LIMIT_PER_MINUTE = tostring(var.ask_rate_limit_per_minute)
    ASK_JUDGE_SAMPLE_RATE     = tostring(var.ask_judge_sample_rate)
  }

  backend_secrets = {
    DATABASE_URL      = google_secret_manager_secret.database_url.secret_id
    ANTHROPIC_API_KEY = google_secret_manager_secret.anthropic_api_key.secret_id
  }

  # The browser calls the API from the frontend's origin -- and, for a
  # private deployment reached through `gcloud run services proxy`, from
  # localhost.
  cors_origins = var.public ? [local.web_url] : [local.web_url, "http://localhost:3000"]
}

# ---- migrations ---------------------------------------------------------------------

# `alembic upgrade head`, run by Terraform itself: a new image tag changes
# run_execution_token, which starts an execution and waits for it to
# succeed -- so the API and worker below never start against a schema
# older than their code.
resource "google_cloud_run_v2_job" "migrate" {
  count = local.deploy_backend ? 1 : 0

  name                = "${var.name}-migrate"
  location            = var.region
  deletion_protection = false
  run_execution_token = substr(sha256(local.backend_image), 0, 16)

  template {
    template {
      service_account = google_service_account.backend.email
      max_retries     = 1
      timeout         = "600s"

      containers {
        image   = local.backend_image
        command = ["alembic", "upgrade", "head"]

        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.db.connection_name]
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.backend_sql,
    google_secret_manager_secret_iam_member.backend,
    google_secret_manager_secret_version.database_url,
    google_sql_database.docpilot,
  ]
}

# ---- API ------------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  count = local.deploy_backend ? 1 : 0

  name                 = local.api_name
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  deletion_protection  = false
  invoker_iam_disabled = var.public

  template {
    service_account                  = google_service_account.backend.email
    timeout                          = "300s"
    max_instance_request_concurrency = 40
    # Second generation: full Linux compatibility, which the Cloud Storage
    # FUSE mount for uploads runs on.
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = 0
      max_instance_count = var.api_max_instances
    }

    containers {
      image = local.backend_image

      ports {
        container_port = 8000
      }

      resources {
        # The embedding model (app/retrieval) loads into memory, and its
        # first download lands in the in-memory /tmp.
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = merge(local.backend_env, { CORS_ORIGINS = jsonencode(local.cors_origins) })
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.backend_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds    = 5
        failure_threshold = 24
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      volume_mounts {
        name       = "uploads"
        mount_path = local.uploads_dir
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.db.connection_name]
      }
    }

    volumes {
      name = "uploads"
      gcs {
        bucket    = google_storage_bucket.uploads.name
        read_only = false
      }
    }
  }

  depends_on = [
    google_cloud_run_v2_job.migrate,
    google_storage_bucket_iam_member.backend_uploads,
  ]
}

# ---- worker -------------------------------------------------------------------------

# The Postgres job queue's consumer (app/worker.py): extraction, indexing
# and grading. A worker pool, not a service -- it pulls work and serves
# no requests -- with exactly one instance, the design the worker assumes.
# The reclaim lease covers the moment a rollout overlaps old and new.
resource "google_cloud_run_v2_worker_pool" "worker" {
  count = local.deploy_backend ? 1 : 0

  name                = "${var.name}-worker"
  location            = var.region
  deletion_protection = false

  scaling {
    scaling_mode          = "MANUAL"
    manual_instance_count = 1
  }

  template {
    service_account = google_service_account.backend.email

    containers {
      image   = local.backend_image
      command = ["python", "-m", "app.worker"]

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }

      dynamic "env" {
        for_each = merge(local.backend_env, {
          WORKER_RECLAIM_AFTER_SECONDS = tostring(var.worker_reclaim_after_seconds)
        })
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.backend_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      volume_mounts {
        name       = "uploads"
        mount_path = local.uploads_dir
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.db.connection_name]
      }
    }

    volumes {
      name = "uploads"
      gcs {
        bucket    = google_storage_bucket.uploads.name
        read_only = false
      }
    }
  }

  depends_on = [
    google_cloud_run_v2_job.migrate,
    google_storage_bucket_iam_member.backend_uploads,
  ]
}

# ---- frontend ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "web" {
  count = local.deploy_frontend ? 1 : 0

  name                 = local.web_name
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  deletion_protection  = false
  invoker_iam_disabled = var.public

  template {
    service_account = google_service_account.frontend.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = local.frontend_image

      ports {
        container_port = 3000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }
    }
  }
}
