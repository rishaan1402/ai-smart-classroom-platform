"""
Configuration — default config + YAML loader + hardware validation.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    # YOLO / inference
    "yolo_model": "yolo11n-pose.pt",
    "yolo_confidence": 0.5,
    "yolo_iou": 0.45,
    "tracker_config": "botsort_custom.yaml",

    # InferencePool (post-MVP)
    "max_batch_size": 8,
    "batch_timeout_ms": 0,  # 0 = disabled; enable when camera_count > 4

    # Camera
    "rtsp_timeout_us": 5_000_000,  # 5 seconds in microseconds
    "camera_reconnect_base_s": 2.0,
    "camera_reconnect_max_s": 30.0,

    # Ring buffer
    "ring_buffer_frames": 900,   # 30s @ 30fps
    "ring_buffer_jpeg_quality": 70,

    # Redis
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_password": "",
    "redis_db": 0,
    # No stream_maxlen — streams are unbounded; StreamTrimmer manages eviction

    # Database
    "db_url": "postgresql+asyncpg://sentinel:sentinel@localhost/sentinel",
    "engine_db_pool_size": 3,

    # MinIO / storage
    "storage_backend": "local",  # "local" or "minio"
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "",
    "minio_secret_key": "",
    "minio_snapshots_bucket": "sentinel-snapshots",
    "minio_recordings_bucket": "sentinel-recordings",
    "storage_local_dir": "data",

    # Workers
    "llm_model": "qwen3:1.7b",
    "vlm_model": "qwen2.5vl:3b",
    "ollama_url": "http://localhost:11434",
    "llm_timeout_s": 10,
    "vlm_timeout_s": 15,
    "stream_trimmer_interval_s": 5,
    "stream_pending_max_age_s": 300,  # 5 minutes before forced persistence

    # Face recognition
    "face_enabled": False,
    "face_model": "buffalo_l",
    "face_similarity_threshold": 0.6,
    "face_check_cooldown_s": 5.0,
    "face_min_crop_size": 64,

    # Gatekeeper
    "hysteresis_frames": 3,
    "fall_confirmation_frames": 5,
    "cooldowns": {
        "FALL_DETECTED": 10.0,
        "RUNNING_IN_ZONE": 5.0,
        "HANDS_RAISED": 3.0,
        "AGGRESSIVE_BEHAVIOR": 15.0,
        "UNKNOWN_PERSON": 30.0,
        "LOITERING": 60.0,
        "ZONE_INTRUSION": 10.0,
    },
    "loitering_threshold_s": 30.0,

    # Event retention
    "event_retention_days": 365,

    # Tenant
    "tenant_id": "default",

    # Logging
    "log_dir": "data/logs",
    "log_level": "INFO",
}

# Hardware throughput estimates (total fps across all cameras)
HARDWARE_FPS_ESTIMATES = {
    "cuda_high": 120,   # RTX 4080+
    "cuda_mid": 66,     # RTX 3060 class
    "mps": 30,
    "cpu": 10,
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load config from YAML file and deep-merge with defaults.
    Missing keys fall back to DEFAULT_CONFIG values.

    Args:
        path: Path to app_config.yaml, or None to use defaults only.

    Returns:
        Merged configuration dict.
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    if path is None:
        return cfg

    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {path} — using defaults")
        return cfg

    try:
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}

        # Deep merge: user values override defaults, missing keys keep defaults
        _deep_merge(cfg, user_cfg)
        logger.info(f"Loaded config from {path}")
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e} — using defaults")

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge of override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def validate_hardware(device: str, camera_count: int, target_fps: int) -> None:
    """
    Log a warning if hardware likely can't sustain the requested camera × fps load.

    Args:
        device: "cuda", "mps", or "cpu"
        camera_count: Number of cameras configured
        target_fps: Target frames per second per camera
    """
    total_required = camera_count * target_fps

    if device == "cuda":
        # Assume mid-tier GPU unless we can query VRAM
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            key = "cuda_high" if vram_gb >= 16 else "cuda_mid"
        except Exception:
            key = "cuda_mid"
    else:
        key = device

    estimated = HARDWARE_FPS_ESTIMATES.get(key, 10)
    effective_per_cam = estimated / max(camera_count, 1)

    if total_required > estimated:
        logger.warning(
            f"Hardware ({key}) estimated throughput: {estimated} fps total. "
            f"Requested: {camera_count} cameras × {target_fps} fps = {total_required} fps. "
            f"Effective FPS per camera: ~{effective_per_cam:.1f}. "
            f"Consider reducing camera count or resolution, or add InferencePool (Phase 1c)."
        )
    else:
        logger.info(
            f"Hardware check passed: {device} estimated {estimated} fps total, "
            f"{camera_count} cameras at {target_fps} fps = {total_required} fps required."
        )
