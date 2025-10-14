# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-01-01
### Added
- Production container based on `python:3.13-slim` running Gunicorn with Uvicorn workers.
- Health endpoint `/healthz` and Docker healthcheck.
- Multi-stage Dockerfile, non-root user, and start/entrypoint scripts.
- Example Docker Compose and Kubernetes manifests (Deployment, Service, HPA).
- Release workflow to build and push images to GHCR on `v*` tags.
- Deployment and operations documentation including SSE proxying and telemetry.

