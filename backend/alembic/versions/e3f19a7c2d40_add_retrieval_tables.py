"""add retrieval tables: document_pages, document_chunks (pgvector), jobs.kind

Revision ID: e3f19a7c2d40
Revises: b7a91c4de2f0
Create Date: 2026-09-29 00:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e3f19a7c2d40'
down_revision: Union[str, Sequence[str], None] = 'b7a91c4de2f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Mirrors app.models.EMBEDDING_DIM at the time this migration was written.
# Deliberately a literal, not an import: a migration must keep describing
# the schema it created even if the app constant later changes.
EMBEDDING_DIM = 384


def upgrade() -> None:
    """Upgrade schema."""
    # Requires the pgvector extension to be installed in the Postgres
    # server (docker-compose.yml and CI use the pgvector/pgvector:pg16
    # image, a drop-in superset of postgres:16).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        'jobs',
        sa.Column('kind', sa.String(), nullable=False, server_default='extract'),
    )

    op.create_table(
        'document_pages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('model', sa.String(), nullable=True),
        sa.Column('prompt_version', sa.String(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'document_id', 'page_number', name='uq_document_pages_document_page'
        ),
    )

    op.create_table(
        'document_chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('page_id', sa.UUID(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('char_start', sa.Integer(), nullable=False),
        sa.Column('char_end', sa.Integer(), nullable=False),
        sa.Column('embedding', Vector(EMBEDDING_DIM), nullable=False),
        sa.Column('embedding_model', sa.String(), nullable=False),
        sa.Column(
            'tsv',
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(context, '') || ' ' || text)",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.ForeignKeyConstraint(['page_id'], ['document_pages.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_document_chunks_document_id', 'document_chunks', ['document_id']
    )
    op.create_index(
        'ix_document_chunks_embedding_hnsw',
        'document_chunks',
        ['embedding'],
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )
    op.create_index(
        'ix_document_chunks_tsv', 'document_chunks', ['tsv'], postgresql_using='gin'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_document_chunks_tsv', table_name='document_chunks')
    op.drop_index('ix_document_chunks_embedding_hnsw', table_name='document_chunks')
    op.drop_index('ix_document_chunks_document_id', table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_table('document_pages')
    op.drop_column('jobs', 'kind')
    # The vector extension itself is left installed: dropping it is a
    # server-level change other databases/schemas may depend on, and
    # leaving it is harmless.
