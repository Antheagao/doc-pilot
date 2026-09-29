"""add ask_runs, and judge jobs that point at them

Revision ID: b9e4c1d7f302
Revises: a4d8e2f6b901
Create Date: 2026-09-29 07:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b9e4c1d7f302'
down_revision: Union[str, Sequence[str], None] = 'a4d8e2f6b901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'ask_runs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('tool_calls', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=False),
        sa.Column('steps', sa.Integer(), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('prompt_version', sa.String(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('cache_read_input_tokens', sa.Integer(), nullable=False),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('refusal_category', sa.String(), nullable=True),
        sa.Column('unresolved_citations', sa.Integer(), nullable=False),
        sa.Column('trace_id', sa.String(), nullable=True),
        sa.Column('traceparent', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('feedback', sa.String(), nullable=True),
        sa.Column('feedback_note', sa.Text(), nullable=True),
        sa.Column('feedback_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('judge_sampled', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('judge_grounded', sa.Boolean(), nullable=True),
        sa.Column('judge_answers_question', sa.Boolean(), nullable=True),
        sa.Column('judge_claims', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('judge_explanation', sa.Text(), nullable=True),
        sa.Column('judge_model', sa.String(), nullable=True),
        sa.Column('judge_prompt_version', sa.String(), nullable=True),
        sa.Column('judge_cost_usd', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('judge_error', sa.Text(), nullable=True),
        sa.Column('judged_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("feedback IN ('up', 'down')", name='ck_ask_runs_feedback'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_ask_runs_created_at', 'ask_runs', ['created_at'], unique=False)

    op.add_column('jobs', sa.Column('ask_run_id', sa.UUID(), nullable=True))
    op.create_foreign_key(
        'jobs_ask_run_id_fkey', 'jobs', 'ask_runs', ['ask_run_id'], ['id'], ondelete='CASCADE'
    )
    op.alter_column('jobs', 'document_id', existing_type=sa.UUID(), nullable=True)
    op.create_check_constraint(
        'ck_jobs_has_target', 'jobs', 'document_id IS NOT NULL OR ask_run_id IS NOT NULL'
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Judge jobs have no document; they can't survive document_id going
    # back to NOT NULL.
    op.execute("DELETE FROM jobs WHERE document_id IS NULL")
    op.drop_constraint('ck_jobs_has_target', 'jobs', type_='check')
    op.alter_column('jobs', 'document_id', existing_type=sa.UUID(), nullable=False)
    op.drop_constraint('jobs_ask_run_id_fkey', 'jobs', type_='foreignkey')
    op.drop_column('jobs', 'ask_run_id')
    op.drop_index('ix_ask_runs_created_at', table_name='ask_runs')
    op.drop_table('ask_runs')
