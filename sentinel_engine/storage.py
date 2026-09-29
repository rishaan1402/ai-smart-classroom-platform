"""
StorageBackend — abstract interface for snapshot/recording storage.

Implementations:
- LocalStorage: writes to local filesystem (dev/single-node)
- MinIOStorage: writes to MinIO/S3 with tenant-prefix isolation

Switch via config: storage_backend = "local" | "minio"

MinIO uses two shared buckets (not per-tenant):
  sentinel-snapshots/{tenant_id}/{camera_id}/{date}/{filename}
  sentinel-recordings/{tenant_id}/{camera_id}/{date}/{filename}
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Save data and return the storage path/URL."""
        ...

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read data by key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data by key."""
        ...

    @abstractmethod
    def url(self, key: str) -> str:
        """Return accessible URL for a stored object."""
        ...


class LocalStorage(StorageBackend):
    """Writes to local filesystem. For dev and single-node deployments."""

    def __init__(self, base_dir: str = "data") -> None:
        self._base = Path(base_dir)
        (self._base / "snapshots").mkdir(parents=True, exist_ok=True)
        (self._base / "recordings").mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: bytes) -> str:
        path = self._base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def read(self, key: str) -> bytes:
        return (self._base / key).read_bytes()

    def delete(self, key: str) -> None:
        p = self._base / key
        if p.exists():
            p.unlink()

    def url(self, key: str) -> str:
        return str(self._base / key)


class MinIOStorage(StorageBackend):
    """
    Writes to MinIO using two shared buckets with tenant-prefix isolation.

    s3://sentinel-snapshots/{tenant_id}/...
    s3://sentinel-recordings/{tenant_id}/...

    Uses prefix-based isolation (not per-tenant buckets) to avoid S3 bucket limits.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        snapshots_bucket: str = "sentinel-snapshots",
        recordings_bucket: str = "sentinel-recordings",
        secure: bool = False,
    ) -> None:
        try:
            import boto3
            from botocore.client import Config
            self._s3 = boto3.client(
                "s3",
                endpoint_url=f"{'https' if secure else 'http'}://{endpoint}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            self._snapshots_bucket = snapshots_bucket
            self._recordings_bucket = recordings_bucket
            self._ensure_buckets()
        except ImportError:
            raise RuntimeError("boto3 not installed: pip install boto3")

    def _ensure_buckets(self) -> None:
        for bucket in (self._snapshots_bucket, self._recordings_bucket):
            try:
                self._s3.head_bucket(Bucket=bucket)
            except Exception:
                try:
                    self._s3.create_bucket(Bucket=bucket)
                    logger.info(f"Created MinIO bucket: {bucket}")
                except Exception as e:
                    logger.error(f"Failed to create bucket {bucket}: {e}")

    def _bucket_for_key(self, key: str) -> str:
        if "recordings" in key or key.endswith(".mp4"):
            return self._recordings_bucket
        return self._snapshots_bucket

    def save(self, key: str, data: bytes) -> str:
        bucket = self._bucket_for_key(key)
        self._s3.put_object(Bucket=bucket, Key=key, Body=data)
        return f"s3://{bucket}/{key}"

    def read(self, key: str) -> bytes:
        bucket = self._bucket_for_key(key)
        resp = self._s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def delete(self, key: str) -> None:
        bucket = self._bucket_for_key(key)
        self._s3.delete_object(Bucket=bucket, Key=key)

    def url(self, key: str) -> str:
        bucket = self._bucket_for_key(key)
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )


def create_storage(cfg: dict) -> StorageBackend:
    """Factory: create storage backend from config."""
    backend = cfg.get("storage_backend", "local")
    if backend == "minio":
        return MinIOStorage(
            endpoint=cfg.get("minio_endpoint", "localhost:9000"),
            access_key=cfg.get("minio_access_key", ""),
            secret_key=cfg.get("minio_secret_key", ""),
            snapshots_bucket=cfg.get("minio_snapshots_bucket", "sentinel-snapshots"),
            recordings_bucket=cfg.get("minio_recordings_bucket", "sentinel-recordings"),
        )
    return LocalStorage(base_dir=cfg.get("storage_local_dir", "data"))
