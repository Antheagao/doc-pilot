"""add reconciliations and extracted_fields.review_reason

Revision ID: e8c3a6f1b254
Revises: d2a7b5c9e013
Create Date: 2026-10-11 14:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e8c3a6f1b254'
down_revision: str | Sequence[str] | None = 'd2a7b5c9e013'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('extracted_fields', sa.Column('review_reason', sa.String(), nullable=True))
    # Every field flagged so far was flagged for low confidence, reviewed or not:
    # a resolved field keeps the needs_review it had, so the backfill covers it
    # and the matching CHECK below holds for existing rows. The arithmetic
    # reasons arrive with scripts/reconcile_all.py, which recomputes routing.
    op.execute("UPDATE extracted_fields SET review_reason = 'confidence' WHERE needs_review")
    op.create_check_constraint(
        'ck_extracted_fields_review_reason',
        'extracted_fields',
        "review_reason ~ '^(confidence|reconciliation:.+)$'",
    )
    op.create_check_constraint(
        'ck_extracted_fields_review_reason_matches',
        'extracted_fields',
        'needs_review = (review_reason IS NOT NULL)',
    )
    # No index on status: three values over one row per extraction, and the
    # unique extraction_id serves every join a report query starts from.
    op.create_table(
        'reconciliations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('extraction_id', sa.UUID(), nullable=False),
        sa.Column('rules_version', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('report', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.CheckConstraint("status IN ('pass', 'fail', 'skipped')", name='ck_reconciliations_status'),
        sa.ForeignKeyConstraint(['extraction_id'], ['extractions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('extraction_id', name='uq_reconciliations_extraction_id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('reconciliations')
    op.drop_constraint('ck_extracted_fields_review_reason_matches', 'extracted_fields', type_='check')
    op.drop_constraint('ck_extracted_fields_review_reason', 'extracted_fields', type_='check')
    op.drop_column('extracted_fields', 'review_reason')
