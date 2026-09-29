"""
CredentialStore — camera credential management.

Credentials encrypted with Fernet (symmetric AES-128) in Python application
memory. Encrypted bytes stored in PostgreSQL camera_credentials table.

The master key (CREDS_MASTER_KEY) NEVER touches PostgreSQL — this eliminates
the session-variable attack surface where a SQL injection could read the key
via SELECT current_setting('app.creds_master_key').

PostgreSQL stores only the opaque encrypted bytes. Python decrypts.

For MVP, also supports a simpler env-var based fallback when no DB connection
is available (dev/testing).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class CredentialStore:
    """
    Camera credential manager using Fernet symmetric encryption.

    Usage:
        store = CredentialStore.from_env()
        creds = store.get_env("cam_01")        # env var fallback (dev)
        store.set(camera_id, "user:pass", db)  # save encrypted to PostgreSQL
        creds = store.get(camera_id, db)        # load + decrypt from PostgreSQL
    """

    def __init__(self, master_key: bytes) -> None:
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(master_key)
        except ImportError:
            raise RuntimeError("cryptography not installed: pip install cryptography")
        except Exception as e:
            raise ValueError(f"Invalid CREDS_MASTER_KEY: {e}")

    @classmethod
    def from_env(cls) -> "CredentialStore":
        """
        Create CredentialStore from CREDS_MASTER_KEY environment variable.

        The key must be a valid Fernet key (32 url-safe base64-encoded bytes).
        Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        """
        key_str = os.environ.get("CREDS_MASTER_KEY", "")
        if not key_str:
            raise EnvironmentError(
                "CREDS_MASTER_KEY environment variable not set. "
                "Generate a key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        return cls(key_str.encode())

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt credential string to bytes for PostgreSQL storage."""
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt bytes from PostgreSQL to plaintext credential."""
        return self._fernet.decrypt(ciphertext).decode()

    def get_env(self, camera_id: str) -> Optional[str]:
        """
        Get credential from environment variable (dev fallback).

        Env var format: SENTINEL_CRED_{CAMERA_ID_UPPER} = "username:password"
        e.g. SENTINEL_CRED_CAM_01 = "admin:secret123"
        """
        env_key = f"SENTINEL_CRED_{camera_id.upper().replace('-', '_').replace('/', '_')}"
        cred = os.environ.get(env_key)
        if cred:
            logger.debug(f"Loaded credential for {camera_id} from env var {env_key}")
        return cred

    def build_rtsp_url(self, host: str, path: str, creds: str) -> str:
        """
        Construct RTSP URL with embedded credentials.

        Args:
            host: Camera host (e.g. "192.168.1.10:554")
            path: RTSP path (e.g. "/h264/ch1/main/av_stream")
            creds: "username:password"

        Returns:
            rtsp://username:password@host/path
        """
        return f"rtsp://{creds}@{host}{path}"

    # ------------------------------------------------------------------
    # PostgreSQL integration (requires asyncpg or psycopg2 session)
    # ------------------------------------------------------------------

    def get(self, camera_id: str, db_session) -> Optional[str]:
        """
        Retrieve and decrypt camera credential from PostgreSQL.

        The SQL function get_credential_enc(cam_id) returns encrypted bytes.
        Tenant isolation enforced by SET LOCAL app.tenant_id in the session.
        The encryption key never appears in any SQL query.

        Args:
            camera_id: UUID string
            db_session: SQLAlchemy session with tenant SET LOCAL already applied

        Returns:
            Plaintext "user:pass" or None if not found
        """
        try:
            from sqlalchemy import text
            row = db_session.execute(
                text("SELECT get_credential_enc(:cam_id)"),
                {"cam_id": camera_id},
            ).scalar()
            if row is None:
                return None
            return self.decrypt(bytes(row))
        except Exception as e:
            logger.error(f"Failed to retrieve credential for {camera_id}: {e}")
            return None

    def set(self, camera_id: str, credential: str, db_session) -> None:
        """
        Encrypt and store camera credential in PostgreSQL.

        Args:
            camera_id: UUID string
            credential: Plaintext "user:pass"
            db_session: SQLAlchemy session
        """
        enc = self.encrypt(credential)
        try:
            from sqlalchemy import text
            db_session.execute(
                text("""
                    INSERT INTO camera_credentials (camera_id, credential_enc)
                    VALUES (:cam_id, :enc)
                    ON CONFLICT (camera_id)
                    DO UPDATE SET credential_enc = EXCLUDED.credential_enc,
                                  rotated_at = NOW()
                """),
                {"cam_id": camera_id, "enc": enc},
            )
        except Exception as e:
            logger.error(f"Failed to store credential for {camera_id}: {e}")
            raise
