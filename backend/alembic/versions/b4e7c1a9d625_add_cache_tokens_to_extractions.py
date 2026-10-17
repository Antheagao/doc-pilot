"""record prompt-cache token counts on extractions

Revision ID: b4e7c1a9d625
Revises: a7d2f4c9b108
Create Date: 2026-10-16 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b4e7c1a9d625'
down_revision: str | Sequence[str] | None = 'a7d2f4c9b108'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # input_tokens counts only the uncached remainder of a call, so a cached
    # call's cost is not recoverable from it alone. Rows written before caching
    # existed correctly hold 0 for both.
    op.add_column(
        'extractions',
        sa.Column('cache_creation_input_tokens', sa.Integer(), server_default='0', nullable=False),
    )
    op.add_column(
        'extractions',
        sa.Column('cache_read_input_tokens', sa.Integer(), server_default='0', nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('extractions', 'cache_read_input_tokens')
    op.drop_column('extractions', 'cache_creation_input_tokens')
