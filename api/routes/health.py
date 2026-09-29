"""
Health + Prometheus metrics endpoints.

GET /api/health: overall system health + VRAM pressure + Redis depth
GET /metrics: Prometheus scrape target
"""

import os
from typing import Any

import redis as redis_lib
import requests
from fastapi import APIRouter, Response, status

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health():
    result: dict[str, Any] = {"status": "healthy"}

    # Ollama
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=2)
        result["ollama_ready"] = r.status_code == 200
    except Exception:
        result["ollama_ready"] = False

    # GPU VRAM pressure
    vram = _get_vram_pressure()
    result["gpu"] = {"vram_pressure": vram["level"], "used_mb": vram["used_mb"]}

    # Redis
    try:
        rc = _redis_client()
        info = rc.info("memory")
        xlen = 0
        try:
            tenant_id = os.getenv("DEFAULT_TENANT_ID", "default")
            xlen = rc.xlen(f"sentinel:events:{tenant_id}")
        except Exception:
            pass
        result["redis"] = {
            "xlen_events": xlen,
            "memory_used_mb": int(info.get("used_memory", 0)) // (1024 * 1024),
            "memory_max_mb": int(info.get("maxmemory", 0)) // (1024 * 1024) or None,
        }
    except Exception:
        result["redis"] = {"status": "unreachable"}
        result["status"] = "degraded"

    # Camera canary status
    try:
        rc = _redis_client()
        cameras = []
        for key in rc.scan_iter(match="canary:*:status", count=100):
            key_str = key.decode() if isinstance(key, bytes) else key
            cam_id = key_str.split(":")[1]
            st = rc.get(key)
            cameras.append({
                "id": cam_id,
                "canary": st.decode() if st else "unknown",
            })
        result["cameras"] = cameras
    except Exception:
        result["cameras"] = []

    # Return 503 if VRAM critical
    if vram["level"] == "critical":
        return Response(
            content=str(result),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )

    return result


@router.get("/metrics")
def metrics():
    """Prometheus text format metrics."""
    lines = []

    try:
        rc = _redis_client()
        tenant_id = os.getenv("DEFAULT_TENANT_ID", "default")
        for stream in ["events", "vlm", "face", "log", "dlq:failed"]:
            key = f"sentinel:{stream}:{tenant_id}"
            try:
                xlen = rc.xlen(key)
                lines.append(f'sentinel_stream_depth{{stream="{stream}"}} {xlen}')
            except Exception:
                pass

        info = rc.info("memory")
        lines.append(f'sentinel_redis_memory_used_bytes {info.get("used_memory", 0)}')
        maxmem = info.get("maxmemory", 0)
        if maxmem:
            pct = (info.get("used_memory", 0) / maxmem) * 100
            lines.append(f'sentinel_redis_memory_pressure_pct {pct:.2f}')
    except Exception:
        pass

    # VRAM
    vram = _get_vram_pressure()
    lines.append(f'sentinel_vram_used_bytes {vram["used_bytes"]}')

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


def _redis_client() -> redis_lib.Redis:
    return redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        db=0,
    )


def _get_vram_pressure() -> dict:
    """Returns {level: low|medium|high|critical, used_mb, used_bytes}."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"level": "n/a", "used_mb": 0, "used_bytes": 0}

        used = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        pct = (used / total) * 100

        if pct > 95:
            level = "critical"
        elif pct > 80:
            level = "high"
        elif pct > 60:
            level = "medium"
        else:
            level = "low"

        return {"level": level, "used_mb": used // (1024 * 1024), "used_bytes": used}
    except Exception:
        return {"level": "n/a", "used_mb": 0, "used_bytes": 0}
