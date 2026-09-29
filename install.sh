#!/usr/bin/env bash
# SentinelCore — one-click installer
# Tested on Ubuntu 22.04 and 24.04 only
set -euo pipefail

log() { echo "[install] $*"; }
die() { echo "[install] ERROR: $*" >&2; exit 1; }

# 1. Check Docker
log "Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
    die "Docker not installed. See https://docs.docker.com/engine/install/"
fi
if ! docker compose version >/dev/null 2>&1; then
    die "Docker Compose plugin not installed."
fi

# 2. Set up .env
if [ ! -f .env ]; then
    log "Creating .env from template..."
    cp .env.example .env

    # Generate cryptographic keys
    log "Generating security keys..."
    CREDS_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null \
        || docker run --rm python:3.11-slim sh -c "pip install -q cryptography && python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
    FACE_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null \
        || docker run --rm python:3.11-slim sh -c "pip install -q cryptography && python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
    REDIS_PW=$(openssl rand -hex 24)
    JWT_SECRET=$(openssl rand -hex 32)
    DB_PW=$(openssl rand -hex 24)
    MINIO_AK=$(openssl rand -hex 12)
    MINIO_SK=$(openssl rand -hex 24)

    sed -i.bak "s|CREDS_MASTER_KEY=.*|CREDS_MASTER_KEY=${CREDS_KEY}|" .env
    sed -i.bak "s|FACE_ENCRYPTION_KEY=.*|FACE_ENCRYPTION_KEY=${FACE_KEY}|" .env
    sed -i.bak "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=${REDIS_PW}|" .env
    sed -i.bak "s|MINIO_ACCESS_KEY=.*|MINIO_ACCESS_KEY=${MINIO_AK}|" .env
    sed -i.bak "s|MINIO_SECRET_KEY=.*|MINIO_SECRET_KEY=${MINIO_SK}|" .env
    {
        echo "DB_PASSWORD=${DB_PW}"
        echo "JWT_SECRET=${JWT_SECRET}"
    } >> .env
    rm -f .env.bak
    log ".env configured with generated keys"
else
    log ".env already exists — skipping key generation"
fi

# 3. Pull + build
log "Pulling base images..."
docker compose pull postgres redis minio ollama || true
log "Building sentinel images..."
docker compose build

# 4. Start services
log "Starting services..."
docker compose up -d

# 5. Wait for health
log "Waiting for API health check..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
        log "API is healthy!"
        break
    fi
    sleep 2
done

# 6. Pull Ollama models
log "Pulling Ollama models (this may take a while)..."
docker compose exec -T ollama ollama pull qwen3:1.7b || log "WARNING: qwen3:1.7b pull failed"
docker compose exec -T ollama ollama pull qwen2.5vl:3b || log "WARNING: qwen2.5vl:3b pull failed"

log ""
log "================================================"
log "SentinelCore installed successfully!"
log ""
log "  Dashboard:  http://localhost:3000"
log "  API:        http://localhost:8000"
log "  API docs:   http://localhost:8000/docs"
log "  MinIO:      http://localhost:9001"
log ""
log "Check logs:   docker compose logs -f"
log "Stop:         docker compose down"
log "================================================"
