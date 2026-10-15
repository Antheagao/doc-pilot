"""allow the quote review reason on extracted_fields

Downgrading is lossy: a field flagged because its evidence quote disagreed with
its value (`quote:<path>`) is flagged for low confidence instead, since the
older rule has no such reason. The flag itself is kept; only the stated cause is
lost, and upgrading again does not restore it (the next reconciliation refresh
recomputes it).

Revision ID: a7d2f4c9b108
Revises: e8c3a6f1b254
Create Date: 2026-10-15 10:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a7d2f4c9b108'
down_revision: str | Sequence[str] | None = 'e8c3a6f1b254'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = 'ck_extracted_fields_review_reason'
TABLE = 'extracted_fields'
# Copied as literals, never imported from app.models, so a later change to the
# model cannot rewrite what this revision did.
WITH_QUOTE = "review_reason ~ '^(confidence|reconciliation:.+|quote:.+)$'"
WITHOUT_QUOTE = "review_reason ~ '^(confidence|reconciliation:.+)$'"


def upgrade() -> None:
    """Upgrade schema."""
    # A CHECK cannot be altered in place: drop it and create the wider one. No
    # row can violate the wider rule, so no data changes.
    op.drop_constraint(CONSTRAINT, TABLE, type_='check')
    op.create_check_constraint(CONSTRAINT, TABLE, WITH_QUOTE)


def downgrade() -> None:
    """Downgrade schema."""
    # Lossy (see the header): the flag and the review state stay as they are,
    # so ck_extracted_fields_review_reason_matches still holds.
    op.execute("UPDATE extracted_fields SET review_reason = 'confidence' WHERE review_reason LIKE 'quote:%'")
    op.drop_constraint(CONSTRAINT, TABLE, type_='check')
    op.create_check_constraint(CONSTRAINT, TABLE, WITHOUT_QUOTE)
