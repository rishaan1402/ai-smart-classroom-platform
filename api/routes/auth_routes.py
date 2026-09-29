"""
Auth routes — login, register, WS token issuance.
"""

import os
from typing import Optional

import redis as redis_lib
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import (
    create_access_token,
    create_ws_token,
    get_current_user,
    hash_password,
    set_auth_cookie,
    verify_password,
    TokenPayload,
)
from ..deps import get_db
from ..middleware.rate_limit import limiter
from ..models import Tenant, User

router = APIRouter(prefix="/api/auth", tags=["auth"])

_redis: Optional[redis_lib.Redis] = None


def _get_redis() -> redis_lib.Redis:
    global _redis
    if _redis is None:
        _redis = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=0,
            decode_responses=False,
        )
    return _redis


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class TokenResponse(BaseModel):
    ws_token: str


@router.post("/login")
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == body.email)).scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    token = create_access_token(str(user.id), str(user.tenant_id), user.role)
    set_auth_cookie(response, token)
    return {"status": "ok", "role": user.role}


@router.post("/register")
def register(body: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    existing = db.execute(select(User).where(User.email == body.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    # Use default tenant (auto-provisioned on first startup)
    tenant = db.execute(select(Tenant).limit(1)).scalar_one_or_none()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No tenant configured",
        )

    # First user on a tenant automatically gets admin role
    existing_users = db.execute(select(User).where(User.tenant_id == tenant.id).limit(1)).scalar_one_or_none()
    role = "admin" if existing_users is None else "viewer"

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        tenant_id=tenant.id,
        role=role,
    )
    db.add(user)
    db.flush()

    token = create_access_token(str(user.id), str(user.tenant_id), user.role)
    set_auth_cookie(response, token)
    return {"status": "ok", "role": user.role}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "ok"}


@router.get("/ws-token", response_model=TokenResponse)
@limiter.limit("5/minute")
def get_ws_token(request: Request, user: TokenPayload = Depends(get_current_user)):
    """Issue a single-use WebSocket token. Rate limited to 5/min per user."""
    r = _get_redis()
    token = create_ws_token(r, user.sub, user.tenant_id, user.role)
    return TokenResponse(ws_token=token)


@router.get("/me")
def get_me(user: TokenPayload = Depends(get_current_user)):
    return {"user_id": user.sub, "tenant_id": user.tenant_id, "role": user.role}
