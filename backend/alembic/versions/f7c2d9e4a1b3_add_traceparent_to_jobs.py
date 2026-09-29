"""add traceparent to jobs

Revision ID: f7c2d9e4a1b3
Revises: e3f19a7c2d40
Create Date: 2026-09-29 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7c2d9e4a1b3'
down_revision: Union[str, Sequence[str], None] = 'e3f19a7c2d40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('jobs', sa.Column('traceparent', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('jobs', 'traceparent')
