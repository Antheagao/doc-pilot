"""index the timestamps the daily budget filters on

Revision ID: c6f1a2b3d4e5
Revises: b9e4c1d7f302
Create Date: 2026-09-29 08:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c6f1a2b3d4e5'
down_revision: Union[str, Sequence[str], None] = 'b9e4c1d7f302'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index('ix_extractions_created_at', 'extractions', ['created_at'], unique=False)
    op.create_index('ix_document_pages_created_at', 'document_pages', ['created_at'], unique=False)
    op.create_index('ix_ask_runs_judged_at', 'ask_runs', ['judged_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_ask_runs_judged_at', table_name='ask_runs')
    op.drop_index('ix_document_pages_created_at', table_name='document_pages')
    op.drop_index('ix_extractions_created_at', table_name='extractions')
