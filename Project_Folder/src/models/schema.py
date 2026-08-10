"""SQLAlchemy Declarative Models for PostgreSQL database objects."""

from sqlalchemy import BigInteger, Column, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ParentDocument(Base):
    """ORM Model representing the parent_documents table for Hierarchical Small-to-Big RAG."""

    __tablename__ = "parent_documents"

    parent_id = Column(String(128), primary_key=True, doc="Unique identifier for the parent chunk")
    doc_id = Column(String(128), nullable=False, index=True, doc="Parent document correlation ID")
    doc_hash_sha256 = Column(String(64), nullable=False, index=True, doc="SHA-256 hash for deduplication")
    source_name = Column(String(255), nullable=False, index=True, doc="Source file name or URL")
    content = Column(Text, nullable=False, doc="Full textual content of the parent section")
    metadata_json = Column("metadata", JSONB, nullable=False, default=dict, doc="JSON metadata payload")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<ParentDocument(parent_id='{self.parent_id}', source='{self.source_name}')>"


class ConversationHistory(Base):
    """ORM Model representing conversational history sessions."""

    __tablename__ = "conversation_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True, doc="Client chat session ID")
    role = Column(String(32), nullable=False, doc="Message author role: user | assistant | system")
    content = Column(Text, nullable=False, doc="Message content")
    metadata_json = Column("metadata", JSONB, nullable=False, default=dict, doc="JSON metadata and citations")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<ConversationHistory(session_id='{self.session_id}', role='{self.role}')>"
