"""Initial database schema with pgvector, parent_documents, and conversation_history.

Revision ID: 001_initial
Revises: 
Create Date: 2026-10-12 22:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create parent_documents table
    op.create_table(
        "parent_documents",
        sa.Column("parent_id", sa.String(length=128), nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("doc_hash_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("parent_id"),
    )
    op.create_index("idx_parent_docs_hash", "parent_documents", ["doc_hash_sha256"], unique=False)
    op.create_index("idx_parent_docs_source", "parent_documents", ["source_name"], unique=False)

    # 3. Create conversation_history table
    op.create_table(
        "conversation_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_conv_session", "conversation_history", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_conv_session", table_name="conversation_history")
    op.drop_table("conversation_history")
    op.drop_index("idx_parent_docs_source", table_name="parent_documents")
    op.drop_index("idx_parent_docs_hash", table_name="parent_documents")
    op.drop_table("parent_documents")
