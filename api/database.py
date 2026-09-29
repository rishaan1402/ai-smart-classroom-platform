"""
Database engine + session factory.

Two pools:
- API: pool_size=10, max_overflow=20 (handles HTTP + WS)
- Engine: pool_size=3 (sentinel_engine workers — separate process)

Both enforce statement_timeout to prevent runaway queries.
"""

import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinel:sentinel@localhost:5432/sentinel",
)

# API connection pool
_api_engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    connect_args={"options": "-c statement_timeout=30000"},  # 30s
)

# Engine connection pool (used by sentinel_engine workers)
ENGINE_DATABASE_URL = os.getenv("ENGINE_DATABASE_URL", DATABASE_URL)
_engine_engine = create_engine(
    ENGINE_DATABASE_URL,
    pool_size=3,
    max_overflow=2,
    pool_pre_ping=True,
    connect_args={"options": "-c statement_timeout=60000"},  # 60s
)

ApiSession = sessionmaker(bind=_api_engine)
EngineSession = sessionmaker(bind=_engine_engine)


def get_api_session() -> Session:
    """Create an API database session."""
    return ApiSession()


def get_engine_session() -> Session:
    """Create an engine database session."""
    return EngineSession()
