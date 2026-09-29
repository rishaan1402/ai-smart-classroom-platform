"""
Authentication — JWT in HttpOnly cookies for REST, first-message protocol for WS.

WebSocket auth flow:
  1. Server sends {"type": "auth_required"}
  2. Client sends {"type": "auth", "token": "...", "last_event_id": "..."} within 5s
  3. Server validates single-use token (Redis SET NX EX 30), sends {"type": "auth_ok"}
  4. If last_event_id: replay missed events from Redis cache, then PostgreSQL fallback
  5. No auth within 5s → close 4401
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis as redis_lib
from fastapi import Cookie, Depends, HTTPException, Request, Response, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import ApiSession
from .models import User

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    sub: str  # user_id
    tenant_id: str
    role: str
    exp: datetime


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, tenant_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_current_user(request: Request) -> TokenPayload:
    """Extract and validate JWT from HttpOnly cookie."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    payload = decode_token(token)
    request.state.tenant_id = payload.tenant_id
    return payload


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "true").lower() == "true",
        samesite="lax",
        max_age=JWT_EXPIRE_MINUTES * 60,
    )


def create_ws_token(redis_client: redis_lib.Redis, user_id: str, tenant_id: str, role: str) -> str:
    """
    Create a single-use WebSocket token (SET NX EX 30).
    Client exchanges this for a WS connection within 30 seconds.
    """
    token = str(uuid.uuid4())
    payload = f"{user_id}:{tenant_id}:{role}"
    redis_client.set(f"ws_token:{token}", payload, nx=True, ex=30)
    return token


def validate_ws_token(redis_client: redis_lib.Redis, token: str) -> Optional[TokenPayload]:
    """
    Validate and consume a single-use WS token.
    Returns None if token is invalid or already used.
    """
    key = f"ws_token:{token}"
    payload = redis_client.get(key)
    if payload is None:
        return None

    # Consume — delete so it can't be reused
    redis_client.delete(key)

    parts = payload.decode().split(":", 2)
    if len(parts) != 3:
        return None

    user_id, tenant_id, role = parts
    return TokenPayload(
        sub=user_id,
        tenant_id=tenant_id,
        role=role,
        exp=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def require_role(*roles: str):
    """Dependency that enforces role-based access."""
    def checker(user: TokenPayload = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(roles)}",
            )
        return user
    return checker
