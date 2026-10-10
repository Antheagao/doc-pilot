"""add review columns to extracted_fields

Revision ID: b7a91c4de2f0
Revises: c538cc803902
Create Date: 2026-07-31 16:40:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7a91c4de2f0'
down_revision: str | Sequence[str] | None = 'c538cc803902'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'extracted_fields',
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'extracted_fields',
        sa.Column('review_action', sa.String(), nullable=True),
    )
    op.add_column(
        'extracted_fields',
        sa.Column(
            'corrected_value',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_index(
        'ix_extracted_fields_review_queue',
        'extracted_fields',
        ['needs_review', 'reviewed_at'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_extracted_fields_review_queue', table_name='extracted_fields')
    op.drop_column('extracted_fields', 'corrected_value')
    op.drop_column('extracted_fields', 'review_action')
    op.drop_column('extracted_fields', 'reviewed_at')
