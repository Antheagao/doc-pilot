output "registry" {
  description = "Where to push images: <registry>/backend:<tag> and <registry>/frontend:<tag>."
  value       = local.registry
}

output "api_url" {
  description = "The API's URL -- build the frontend image with NEXT_PUBLIC_API_URL set to this (or to http://localhost:8000 for a private deployment)."
  value       = local.api_url
}

output "web_url" {
  description = "The frontend's URL."
  value       = local.web_url
}

output "database_connection_name" {
  description = "Cloud SQL connection name, for `cloud-sql-proxy` or the console."
  value       = google_sql_database_instance.db.connection_name
}

output "anthropic_api_key_secret" {
  description = "Add the API key as a version of this secret before deploying the services."
  value       = google_secret_manager_secret.anthropic_api_key.secret_id
}

output "uploads_bucket" {
  value = google_storage_bucket.uploads.name
}
