"""scope every content table to an organization

Revision ID: 127aac687508
Revises: aa652338a4ae
Create Date: 2026-10-25 14:00:00.000000

Every row that exists before this revision was written by the one-tenant app,
so the backfill puts all of it in the default organization.

New rows take their organization from the transaction: the application sets
`docpilot.organization_id` at the start of each one (app/tenancy.py), and the
column default `docpilot_current_organization_id()` reads it. Code never sets
organization_id itself. With no organization set the function returns NULL, so
an insert from a session that forgot to bind one is refused by NOT NULL rather
than filed under the wrong tenant.

The downgrade is lossy: it drops organization_id, so upgrading again files
every organization's rows under the default one.

Each child row is tied to its parent's organization by a composite foreign key
(parent id, organization_id), which needs UNIQUE (id, organization_id) on the
four parents. Each composite foreign key keeps the ON DELETE behavior of the
single-column key it replaces.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '127aac687508'
down_revision: str | Sequence[str] | None = 'aa652338a4ae'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Copied from app.models.DEFAULT_ORGANIZATION_ID, not imported: a migration
# keeps meaning what it meant when it was written.
DEFAULT_ORGANIZATION_ID = '270ad65a-150a-498e-bbb1-584f6d21d86e'

CONTENT_TABLES = (
    'documents',
    'jobs',
    'extractions',
    'extracted_fields',
    'reconciliations',
    'document_pages',
    'document_chunks',
    'ask_runs',
)

# The tables other tables point at, which need UNIQUE (id, organization_id) for
# a composite foreign key to reference.
PARENT_TABLES = ('documents', 'extractions', 'document_pages', 'ask_runs')

# (child, parent, child column, ON DELETE of the single-column key it replaces,
# or None for the default NO ACTION). The old key's name is Postgres's own:
# <child>_<column>_fkey, read from pg_constraint before this was written.
FOREIGN_KEYS = (
    ('jobs', 'documents', 'document_id', None),
    ('jobs', 'ask_runs', 'ask_run_id', 'CASCADE'),
    ('extractions', 'documents', 'document_id', None),
    ('extracted_fields', 'extractions', 'extraction_id', None),
    ('reconciliations', 'extractions', 'extraction_id', 'CASCADE'),
    ('document_pages', 'documents', 'document_id', None),
    ('document_chunks', 'documents', 'document_id', None),
    ('document_chunks', 'document_pages', 'page_id', None),
    ('ask_runs', 'documents', 'document_id', None),
)

# (name, table, columns). The first group serves the organization-scoped lists
# and sums; the second the composite foreign keys, where no existing index
# already leads with the child column (jobs, extractions, and
# document_chunks.page_id had none).
INDEXES = (
    ('ix_documents_organization_id_created_at', 'documents', ['organization_id', 'created_at']),
    ('ix_ask_runs_organization_id_created_at', 'ask_runs', ['organization_id', 'created_at']),
    ('ix_extractions_organization_id_created_at', 'extractions', ['organization_id', 'created_at']),
    ('ix_document_pages_organization_id_created_at', 'document_pages', ['organization_id', 'created_at']),
    ('ix_document_chunks_organization_id', 'document_chunks', ['organization_id']),
    (
        'ix_extracted_fields_organization_id_needs_review_reviewed_at',
        'extracted_fields',
        ['organization_id', 'needs_review', 'reviewed_at'],
    ),
    ('ix_jobs_organization_id', 'jobs', ['organization_id']),
    ('ix_reconciliations_organization_id', 'reconciliations', ['organization_id']),
    ('ix_jobs_document_id_organization_id', 'jobs', ['document_id', 'organization_id']),
    ('ix_jobs_ask_run_id_organization_id', 'jobs', ['ask_run_id', 'organization_id']),
    ('ix_extractions_document_id_organization_id', 'extractions', ['document_id', 'organization_id']),
    ('ix_document_chunks_page_id_organization_id', 'document_chunks', ['page_id', 'organization_id']),
)


def _composite_name(child: str, column: str) -> str:
    return f'{child}_{column}_organization_id_fkey'


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE FUNCTION docpilot_current_organization_id() RETURNS uuid
        LANGUAGE sql STABLE
        AS $$ SELECT nullif(current_setting('docpilot.organization_id', true), '')::uuid $$
        """
    )

    for table in CONTENT_TABLES:
        op.add_column(table, sa.Column('organization_id', sa.UUID(), nullable=True))
        op.execute(f"UPDATE {table} SET organization_id = '{DEFAULT_ORGANIZATION_ID}'")
        # The large-table way to NOT NULL: a validated CHECK lets SET NOT NULL
        # skip its own scan of the table.
        check = f'ck_{table}_organization_id_not_null'
        op.execute(
            f'ALTER TABLE {table} ADD CONSTRAINT {check} CHECK (organization_id IS NOT NULL) NOT VALID'
        )
        op.execute(f'ALTER TABLE {table} VALIDATE CONSTRAINT {check}')
        op.alter_column(table, 'organization_id', nullable=False)
        op.drop_constraint(check, table, type_='check')
        op.execute(
            f'ALTER TABLE {table} ALTER COLUMN organization_id '
            'SET DEFAULT docpilot_current_organization_id()'
        )
        # RESTRICT: deleting an organization's data is an explicit job, never
        # a side effect of deleting the organization.
        fkey = f'{table}_organization_id_fkey'
        op.execute(
            f'ALTER TABLE {table} ADD CONSTRAINT {fkey} FOREIGN KEY (organization_id) '
            'REFERENCES organizations (id) ON DELETE RESTRICT NOT VALID'
        )
        op.execute(f'ALTER TABLE {table} VALIDATE CONSTRAINT {fkey}')

    for table in PARENT_TABLES:
        op.create_unique_constraint(f'uq_{table}_id_organization_id', table, ['id', 'organization_id'])

    # NOT VALID then VALIDATE, so the table stays readable and writable while
    # the existing rows are checked. The old key goes only once the new one holds.
    for child, parent, column, on_delete in FOREIGN_KEYS:
        name = _composite_name(child, column)
        action = f' ON DELETE {on_delete}' if on_delete else ''
        op.execute(
            f'ALTER TABLE {child} ADD CONSTRAINT {name} '
            f'FOREIGN KEY ({column}, organization_id) REFERENCES {parent} (id, organization_id){action} '
            'NOT VALID'
        )
        op.execute(f'ALTER TABLE {child} VALIDATE CONSTRAINT {name}')
        op.drop_constraint(f'{child}_{column}_fkey', child, type_='foreignkey')

    for name, table, columns in INDEXES:
        op.create_index(name, table, columns, unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    for name, table, _columns in reversed(INDEXES):
        op.drop_index(name, table_name=table)

    # The single-column keys come back with the names and ON DELETE behavior
    # they had, before the composite keys that replaced them go.
    for child, parent, column, on_delete in reversed(FOREIGN_KEYS):
        op.create_foreign_key(
            f'{child}_{column}_fkey', child, parent, [column], ['id'], ondelete=on_delete
        )
        op.drop_constraint(_composite_name(child, column), child, type_='foreignkey')

    for table in reversed(PARENT_TABLES):
        op.drop_constraint(f'uq_{table}_id_organization_id', table, type_='unique')

    for table in reversed(CONTENT_TABLES):
        op.drop_constraint(f'{table}_organization_id_fkey', table, type_='foreignkey')
        op.drop_column(table, 'organization_id')

    op.execute('DROP FUNCTION docpilot_current_organization_id()')
