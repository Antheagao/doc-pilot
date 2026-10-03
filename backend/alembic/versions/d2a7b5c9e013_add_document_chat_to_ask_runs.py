"""per-document chat: ask_runs.document_id and conversation_id

Revision ID: d2a7b5c9e013
Revises: c6f1a2b3d4e5
Create Date: 2026-10-03 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd2a7b5c9e013'
down_revision: Union[str, Sequence[str], None] = 'c6f1a2b3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('ask_runs', sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('ask_runs', sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'ask_runs_document_id_fkey', 'ask_runs', 'documents', ['document_id'], ['id']
    )
    op.create_check_constraint(
        'ck_ask_runs_chat_scope', 'ask_runs', '(document_id IS NULL) = (conversation_id IS NULL)'
    )
    op.create_index(
        'ix_ask_runs_conversation', 'ask_runs', ['conversation_id', 'created_at'], unique=False
    )
    op.create_index('ix_ask_runs_document_id', 'ask_runs', ['document_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_ask_runs_document_id', table_name='ask_runs')
    op.drop_index('ix_ask_runs_conversation', table_name='ask_runs')
    op.drop_constraint('ck_ask_runs_chat_scope', 'ask_runs', type_='check')
    op.drop_constraint('ask_runs_document_id_fkey', 'ask_runs', type_='foreignkey')
    op.drop_column('ask_runs', 'conversation_id')
    op.drop_column('ask_runs', 'document_id')
