"""add search_aliases to document_chunks and include it in the tsvector

Revision ID: a4d8e2f6b901
Revises: f7c2d9e4a1b3
Create Date: 2026-09-29 07:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a4d8e2f6b901'
down_revision: Union[str, Sequence[str], None] = 'f7c2d9e4a1b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_TSV = "to_tsvector('english', coalesce(context, '') || ' ' || text)"
NEW_TSV = (
    "to_tsvector('english', coalesce(context, '') || ' ' || text || ' ' "
    "|| coalesce(search_aliases, ''))"
)


def _replace_tsv(expression: str) -> None:
    # A generated column's expression can't be altered in place: drop it
    # (and its GIN index) and add it back, which recomputes every row.
    op.drop_index('ix_document_chunks_tsv', table_name='document_chunks')
    op.drop_column('document_chunks', 'tsv')
    op.add_column(
        'document_chunks',
        sa.Column('tsv', postgresql.TSVECTOR(), sa.Computed(expression, persisted=True), nullable=True),
    )
    op.create_index('ix_document_chunks_tsv', 'document_chunks', ['tsv'], postgresql_using='gin')


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('document_chunks', sa.Column('search_aliases', sa.Text(), nullable=True))
    _replace_tsv(NEW_TSV)
    # Existing chunks have no aliases until re-indexed (scripts/reindex.py
    # --all); their search behaves exactly as before in the meantime.


def downgrade() -> None:
    """Downgrade schema."""
    _replace_tsv(OLD_TSV)
    op.drop_column('document_chunks', 'search_aliases')
