# SentinelCore — Operations Runbook

## Hardware Requirements

| GPU | Max Cameras | Effective FPS | Notes |
|---|---|---|---|
| RTX 4080 / 4090 | 12 | 15 | Ideal configuration |
| RTX 3060 / 3070 | 6 | 10 | Baseline recommended |
| Apple M1/M2 (MPS) | 3 | 8 | Development only |
| CPU only | 2 | 5 | Not recommended |

**Ollama models:**
- `qwen3:1.7b` — ~1GB VRAM (LLM triage)
- `qwen2.5vl:3b` — ~2GB VRAM (VLM image analysis)
- YOLO11n-pose — ~300MB VRAM

Minimum total VRAM: **6GB**. Comfortable on 8GB. 12GB = room for 6+ cameras.

## Quick Operational Commands

```bash
# Tail logs
docker compose logs -f engine
docker compose logs -f api

# Check stream depths
docker compose exec redis redis-cli -a $REDIS_PASSWORD XLEN sentinel:events:default

# Check Redis memory
docker compose exec redis redis-cli -a $REDIS_PASSWORD INFO memory

# Graceful restart
docker compose restart engine

# Run backup
docker compose run --rm sentinel-backup
```

## Troubleshooting

| Symptom | Diagnosis | Action |
|---|---|---|
| VRAM pressure critical (health returns 503) | GPU saturated | Reduce camera count or target_fps |
| `sentinel_event_queue_depth` rising fast | Ollama slow/hung | `docker compose logs ollama`; restart if stuck |
| Redis `maxmemory` error in engine logs | Stream fill — StreamTrimmer can't keep up | Increase `maxmemory` in docker-compose.yml |
| Camera offline > 10 min | RTSP disconnect | Check physical connection; use `POST /api/cameras/{id}/test` |
| `sentinel_stream_skipped_events` > 0 | Ollama too slow | Events persisted without LLM enrichment — this is safe but non-ideal |
| `sentinel_dlq_failed` growing | Ollama model broken | `docker compose exec ollama ollama pull qwen3:8b` |
| All cameras show "canary degraded" | YOLO or model issue | Check GPU drivers; verify `yolo11n-pose.pt` downloaded |
| RTX 3060 with > 6 cameras | Expected FPS degradation | Post-MVP: enable InferencePool (Phase 1c) |
| WebSocket disconnects after 5s | Client not sending auth message | Client must send `{type:"auth",token:"..."}` within 5s |
| "no tenant configured" on register | Default tenant not auto-provisioned | Restart API; check `tenants` table |

## Shutdown Behavior

Graceful shutdown takes up to 20 seconds:
1. CameraWorkers stop capturing (immediate)
2. Ring buffers flushed to `data/ring_buffer_checkpoint/` (< 1s)
3. LLMWorkers drain (up to 5s)
4. VLMWorkers drain (up to 10s)
5. FaceWorkers + EventLoggers drain (up to 5s)

`docker compose stop` uses `stop_grace_period: 25s` to allow full drain.
Pending clips are resumed on next startup via `resume_from_checkpoint()`.

## Security Checklist

- [ ] `CREDS_MASTER_KEY` and `FACE_ENCRYPTION_KEY` are randomly generated (not defaults)
- [ ] `REDIS_PASSWORD` is set and non-default
- [ ] `JWT_SECRET` is set and at least 32 bytes
- [ ] `COOKIE_SECURE=true` in production (requires HTTPS)
- [ ] PostgreSQL `sslmode=require` in production
- [ ] `camera_credentials.credential_enc` is Fernet-encrypted (never plaintext)
- [ ] Face embeddings have `expires_at` set (default 90 days)
- [ ] Backup cron job scheduled on host

## Metrics (Prometheus)

Scrape `GET /metrics`:
- `sentinel_stream_depth{stream="events|vlm|face|log|dlq:failed"}`
- `sentinel_redis_memory_used_bytes`
- `sentinel_redis_memory_pressure_pct`
- `sentinel_vram_used_bytes`

Alert thresholds:
- `sentinel_redis_memory_pressure_pct > 80` → WARNING
- `sentinel_redis_memory_pressure_pct > 95` → CRITICAL
- `sentinel_stream_depth{stream="dlq:failed"} > 100` → WARNING
