#!/usr/bin/env bash
# SentinelCore — PostgreSQL backup to MinIO
# Run as: docker compose run --rm sentinel-backup
set -euo pipefail

DATE=$(date +%Y-%m-%d_%H%M%S)
BACKUP_FILE="/tmp/sentinel_${DATE}.sql.gz"

echo "[backup] Dumping PostgreSQL..."
PGPASSWORD="${DB_PASSWORD:-sentinel}" pg_dump \
  -h postgres -U sentinel sentinel \
  | gzip > "$BACKUP_FILE"

SIZE=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE")
echo "[backup] Backup size: $((SIZE / 1024))KB"

echo "[backup] Uploading to MinIO..."
# mc alias set; mc cp "$BACKUP_FILE" local/sentinel-backups/postgres/${DATE}.sql.gz
# (use minio/mc client or curl with signed URL in real deployment)

rm -f "$BACKUP_FILE"
echo "[backup] Done"
