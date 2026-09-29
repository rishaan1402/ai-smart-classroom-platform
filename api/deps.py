"""
FastAPI dependency injection — database sessions with RLS tenant scoping.

get_db(): for HTTP/WS requests — extracts tenant_id from JWT
get_db_for_worker(): for engine background workers — tenant_id from source event
"""

from typing import Generator

from fastapi import Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import ApiSession, EngineSession


def get_db(request: Request) -> Generator[Session, None, None]:
    """
    Yield a tenant-scoped DB session for API requests.

    Sets `app.tenant_id` via SET LOCAL so PostgreSQL RLS policies
    filter rows automatically. SET LOCAL scopes to the transaction.
    """
    db = ApiSession()
    try:
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            db.execute(text("SET LOCAL app.tenant_id = :tid"), {"tid": str(tenant_id)})
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_for_worker(tenant_id: str) -> Generator[Session, None, None]:
    """
    Yield a tenant-scoped DB session for engine workers.

    Same SET LOCAL pattern but uses the smaller engine pool.
    """
    db = EngineSession()
    try:
        db.execute(text("SET LOCAL app.tenant_id = :tid"), {"tid": tenant_id})
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
