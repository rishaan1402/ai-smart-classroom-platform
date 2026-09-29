"""initial schema with RLS

Revision ID: 0001
Revises:
Create Date: 2026-04-10

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # tenants
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # cameras
    op.create_table(
        "cameras",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("host", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("zone_config", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # camera_credentials (encrypted blobs — Fernet in Python)
    op.create_table(
        "camera_credentials",
        sa.Column(
            "camera_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cameras.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("credential_enc", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("rotated_at", sa.DateTime(timezone=True)),
    )

    # face_embeddings (pgcrypto for DB-level expiry queries)
    op.create_table(
        "face_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("embedding_enc", sa.LargeBinary()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), server_default=sa.text("NOW() + INTERVAL '90 days'")),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("anonymized_at", sa.DateTime(timezone=True)),
    )

    # events
    op.create_table(
        "events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("cameras.id")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("person_id", sa.Integer()),
        sa.Column(
            "face_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_embeddings.id", ondelete="SET NULL"),
        ),
        sa.Column("face_recognition_was_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("enrichment_skipped", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("threat_level", sa.Text()),
        sa.Column("snapshot_path", sa.Text()),
        sa.Column("clip_path", sa.Text()),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_events_tenant_severity_created", "events", ["tenant_id", "severity", sa.text("created_at DESC")])
    op.create_index("ix_events_tenant_trace", "events", ["tenant_id", "trace_id"])

    # users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.Text(), unique=True, nullable=False),
        sa.Column("hashed_password", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default=sa.text("'viewer'")),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # Row-Level Security — tenant_id from SET LOCAL app.tenant_id
    for table in ["events", "cameras", "camera_credentials", "face_embeddings", "users"]:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    op.execute("""
        CREATE POLICY tenant_isolation ON events
          USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
    """)
    op.execute("""
        CREATE POLICY tenant_isolation ON cameras
          USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
    """)
    op.execute("""
        CREATE POLICY tenant_isolation ON camera_credentials
          USING (
            camera_id IN (
              SELECT id FROM cameras
              WHERE tenant_id = current_setting('app.tenant_id', true)::uuid
            )
          )
    """)
    op.execute("""
        CREATE POLICY tenant_isolation ON face_embeddings
          USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
    """)
    op.execute("""
        CREATE POLICY tenant_isolation ON users
          USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
    """)

    # Roles
    op.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sentinel_engine') THEN
            CREATE ROLE sentinel_engine;
          END IF;
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sentinel_api') THEN
            CREATE ROLE sentinel_api;
          END IF;
        END
        $$;
    """)

    # Engine: INSERT events, UPDATE only clip_path, SELECT cameras/credentials
    op.execute("GRANT INSERT ON events TO sentinel_engine")
    op.execute("GRANT UPDATE (clip_path) ON events TO sentinel_engine")
    op.execute("GRANT SELECT ON cameras, camera_credentials TO sentinel_engine")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON face_embeddings TO sentinel_engine")
    # API: full access
    op.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO sentinel_api")

    # get_credential_enc function — explicit tenant check in SECURITY DEFINER
    op.execute("""
        CREATE OR REPLACE FUNCTION get_credential_enc(cam_id UUID) RETURNS BYTEA AS $$
          SELECT cc.credential_enc
          FROM camera_credentials cc
          JOIN cameras c ON cc.camera_id = c.id
          WHERE cc.camera_id = cam_id
            AND c.tenant_id = current_setting('app.tenant_id', true)::uuid;
        $$ LANGUAGE sql SECURITY DEFINER;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS get_credential_enc(UUID)")
    op.drop_table("users")
    op.drop_index("ix_events_tenant_trace", table_name="events")
    op.drop_index("ix_events_tenant_severity_created", table_name="events")
    op.drop_table("events")
    op.drop_table("face_embeddings")
    op.drop_table("camera_credentials")
    op.drop_table("cameras")
    op.drop_table("tenants")
