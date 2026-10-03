# Offline tests of the deployment's logic: providers are mocked, so these
# need no GCP credentials and create nothing (`terraform test`).

mock_provider "google" {
  mock_data "google_project" {
    defaults = {
      number = "123456789012"
    }
  }

  mock_resource "google_sql_database_instance" {
    defaults = {
      connection_name = "demo-project:us-central1:doc-pilot-db"
    }
  }

  mock_resource "google_service_account" {
    defaults = {
      email  = "doc-pilot@demo-project.iam.gserviceaccount.com"
      member = "serviceAccount:doc-pilot@demo-project.iam.gserviceaccount.com"
    }
  }
}

# The random provider runs for real: it is local (no credentials), and
# Terraform can't mock the ephemeral password it generates.

variables {
  project_id = "demo-project"
}

run "first_apply_creates_the_foundation_but_no_containers" {
  command = plan

  assert {
    condition = (
      length(google_cloud_run_v2_job.migrate) == 0 &&
      length(google_cloud_run_v2_service.api) == 0 &&
      length(google_cloud_run_v2_worker_pool.worker) == 0 &&
      length(google_cloud_run_v2_service.web) == 0
    )
    error_message = "Nothing should run before an image tag is set: Cloud Run refuses images that haven't been pushed."
  }

  assert {
    condition     = output.api_url == "https://doc-pilot-api-123456789012.us-central1.run.app"
    error_message = "api_url must be Cloud Run's deterministic URL, so the frontend can be built against it."
  }
}

run "a_private_deployment" {
  command = apply

  variables {
    backend_image_tag  = "v1"
    frontend_image_tag = "v1"
  }

  assert {
    condition     = google_cloud_run_v2_service.api[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/demo-project/doc-pilot/backend:v1"
    error_message = "The API runs the pushed backend image."
  }

  assert {
    condition     = google_cloud_run_v2_worker_pool.worker[0].template[0].containers[0].image == google_cloud_run_v2_service.api[0].template[0].containers[0].image
    error_message = "The worker runs the same backend image as the API."
  }

  assert {
    condition     = google_cloud_run_v2_worker_pool.worker[0].scaling[0].manual_instance_count == 1
    error_message = "Exactly one worker: app/worker.py's reclaim sweep assumes it."
  }

  assert {
    condition     = google_cloud_run_v2_job.migrate[0].template[0].template[0].containers[0].command == tolist(["alembic", "upgrade", "head"])
    error_message = "The migration job runs alembic."
  }

  assert {
    condition     = length(google_cloud_run_v2_job.migrate[0].run_execution_token) == 16
    error_message = "A new image must start (and wait for) a migration execution."
  }

  assert {
    condition     = !google_cloud_run_v2_service.api[0].invoker_iam_disabled && !google_cloud_run_v2_service.web[0].invoker_iam_disabled
    error_message = "Private by default: only roles/run.invoker reaches the services."
  }

  assert {
    condition = jsondecode(one([
      for e in google_cloud_run_v2_service.api[0].template[0].containers[0].env : e.value if e.name == "CORS_ORIGINS"
    ])) == ["https://doc-pilot-web-123456789012.us-central1.run.app", "http://localhost:3000"]
    error_message = "A private deployment allows the frontend's origin and the local proxy's."
  }

  assert {
    condition = one([
      for e in google_cloud_run_v2_worker_pool.worker[0].template[0].containers[0].env : e.value if e.name == "WORKER_RECLAIM_AFTER_SECONDS"
    ]) == "1800"
    error_message = "The worker reclaims with a lease on Cloud Run."
  }

  assert {
    condition = alltrue([
      for e in google_cloud_run_v2_service.api[0].template[0].containers[0].env :
      e.value_source[0].secret_key_ref[0].version == "latest"
      if contains(["DATABASE_URL", "ANTHROPIC_API_KEY"], e.name)
    ])
    error_message = "Credentials come from Secret Manager, never plain env values."
  }

  assert {
    condition = length([
      for e in google_cloud_run_v2_service.api[0].template[0].containers[0].env : e
      if contains(["DATABASE_URL", "ANTHROPIC_API_KEY"], e.name)
    ]) == 2
    error_message = "The API gets both secrets."
  }
}

run "a_public_deployment" {
  command = apply

  variables {
    backend_image_tag  = "v1"
    frontend_image_tag = "v1"
    public             = true
  }

  assert {
    condition     = google_cloud_run_v2_service.api[0].invoker_iam_disabled && google_cloud_run_v2_service.web[0].invoker_iam_disabled
    error_message = "public = true opens both services."
  }

  assert {
    condition = jsondecode(one([
      for e in google_cloud_run_v2_service.api[0].template[0].containers[0].env : e.value if e.name == "CORS_ORIGINS"
    ])) == ["https://doc-pilot-web-123456789012.us-central1.run.app"]
    error_message = "A public deployment allows only the frontend's origin."
  }
}

run "the_backend_can_deploy_without_a_frontend" {
  command = plan

  variables {
    backend_image_tag = "v1"
  }

  assert {
    condition     = length(google_cloud_run_v2_service.api) == 1 && length(google_cloud_run_v2_service.web) == 0
    error_message = "The frontend waits for its own image tag."
  }
}

run "the_budget_cannot_be_negative" {
  command = plan

  variables {
    daily_budget_usd = -1
  }

  expect_failures = [var.daily_budget_usd]
}
