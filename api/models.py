"""
SQLAlchemy ORM models — mirrors PostgreSQL schema from the plan.

RLS is enforced at the DB level; these models define the Python-side schema.
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    config = Column(JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

    cameras = relationship("Camera", back_populates="tenant")
    events = relationship("Event", back_populates="tenant")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    name = Column(Text)
    host = Column(Text, nullable=False)
    path = Column(Text, nullable=False)
    enabled = Column(Boolean, server_default=text("true"))
    zone_config = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

    tenant = relationship("Tenant", back_populates="cameras")
    credential = relationship("CameraCredential", back_populates="camera", uselist=False)
    events = relationship("Event", back_populates="camera")


class CameraCredential(Base):
    __tablename__ = "camera_credentials"

    camera_id = Column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        primary_key=True,
    )
    credential_enc = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    rotated_at = Column(DateTime(timezone=True))

    camera = relationship("Camera", back_populates="credential")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(Text)
    embedding_enc = Column(LargeBinary)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    expires_at = Column(
        DateTime(timezone=True),
        server_default=text("NOW() + INTERVAL '90 days'"),
    )
    deleted_at = Column(DateTime(timezone=True))
    anonymized_at = Column(DateTime(timezone=True))


class Event(Base):
    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    schema_version = Column(Integer, nullable=False, server_default=text("1"))
    trace_id = Column(UUID(as_uuid=True), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"))
    event_type = Column(Text, nullable=False)
    severity = Column(Text, nullable=False)
    person_id = Column(Integer)
    face_id = Column(
        UUID(as_uuid=True),
        ForeignKey("face_embeddings.id", ondelete="SET NULL"),
    )
    face_recognition_was_active = Column(Boolean, nullable=False, server_default=text("false"))
    enrichment_skipped = Column(Boolean, nullable=False, server_default=text("false"))
    threat_level = Column(Text)
    snapshot_path = Column(Text)
    clip_path = Column(Text)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

    tenant = relationship("Tenant", back_populates="events")
    camera = relationship("Camera", back_populates="events")

    __table_args__ = (
        Index("ix_events_tenant_severity_created", "tenant_id", "severity", created_at.desc()),
        Index("ix_events_tenant_trace", "tenant_id", "trace_id"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    email = Column(Text, unique=True, nullable=False)
    hashed_password = Column(Text, nullable=False)
    role = Column(Text, nullable=False, server_default=text("'viewer'"))
    is_active = Column(Boolean, server_default=text("true"))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
