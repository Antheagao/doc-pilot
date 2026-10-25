"""enforce tenant isolation with row-level security

Revision ID: 73a7d8691932
Revises: 127aac687508
Create Date: 2026-10-25 18:00:00.000000

The database layer of tenancy. The application layer (app/tenancy.py) filters
every ORM statement to one organization; this revision makes Postgres refuse
what that layer cannot see (raw text() SQL, a new route that forgets a filter)
by enabling and forcing row-level security on the eight content tables.

Two settings drive it, both transaction-local and both set by the application
at the start of every transaction:
- `docpilot.organization_id`: the tenant. The policy `tenant_isolation` admits
  a row only when its organization_id equals docpilot_current_organization_id(),
  which reads this setting, for reads and as WITH CHECK for writes. With no
  tenant set the function is NULL, so a read finds nothing and a write is refused.
- `docpilot.queue_scope`: `on` only inside app.tenancy.queue_session, the
  worker's claim of the oldest pending job in any organization. Two more
  policies on jobs, `jobs_queue_claim` (SELECT) and `jobs_queue_update` (UPDATE),
  admit every row while it is on. No other table has one, and a queue session can
  neither insert nor delete.

Threat statement: the settings are written by the application, so row-level
security here contains application bugs (a query that forgets its filter, a
text() statement, a route added later), not an attacker who can run arbitrary
SQL as the application's role: that attacker can set the tenant to any
organization. Making the tenant unforgeable (a definer function that derives it
from a verified session token) is a hardening option, not built.

The policies bind only a role that is subject to them. A superuser and a role
with BYPASSRLS skip every policy, and so does a table's owner unless the table
is FORCEd (these are, though the owner here is a superuser and skips them
regardless). So the application must connect as a member of the new group role `docpilot_app`
(created here, NOLOGIN, holder of every grant) through a login role that
`scripts/provision_db_role.py` creates outside Alembic, because a login role and
its password do not belong in a migration. app.db.verify_tenant_isolation
refuses to start a process whose role is a superuser, has BYPASSRLS, or owns
the tables. The identity tables (organizations, users, memberships,
auth_sessions, api_keys) have no policy: a sign-in reads them before any tenant
is known. alembic_version is not granted to the application role.

Row-level security cannot compare a row's old and new values, so the queue's
UPDATE policy alone would let a claim re-file a job under another organization;
UPDATE on jobs is therefore granted by column, and organization_id, document_id,
ask_run_id, kind and traceparent are not among them.

A table added later needs its own GRANT to docpilot_app, and a content table its
own policies; tests/test_rls.py fails until it has them.

docpilot_spend_since(since) is the installation-wide model spend since a time,
for the one daily budget that guards the one Anthropic key. It is SECURITY
DEFINER so it can sum across tenants, which only works because its owner (the
migration role, a superuser) bypasses row-level security. It returns a number,
never a row, and only docpilot_app may execute it. The fixed search_path keeps a
caller from shadowing a table or operator it uses; pg_temp is last so a temporary
table cannot shadow one either.

Roles are cluster-wide while grants belong to one database, and the dev, demo,
and test databases share a cluster. The downgrade therefore removes every grant
and policy of this database and drops docpilot_app only when nothing else uses
it. While another database still grants to it Postgres refuses (2BP01), and
while a login role is still a member of it dropping it would silently strip
that membership (a later upgrade creates a new, empty group, and the login role
would no longer be recognized as the application's); in both cases the role is
kept with a NOTICE.
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '73a7d8691932'
down_revision: str | Sequence[str] | None = '127aac687508'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

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

IDENTITY_TABLES = ('organizations', 'users', 'memberships', 'auth_sessions', 'api_keys')

# Every table the application uses; alembic_version is not one of them. The
# keys are UUIDs, so there are no sequences to grant.
APP_TABLES = (*IDENTITY_TABLES, *CONTENT_TABLES)

# What app/worker.py changes on a job after it is inserted: the claim (state,
# started_at, attempts), completion (state, finished_at), and failure or requeue
# (state, last_error, run_after, finished_at, attempts).
WORKER_UPDATED_JOB_COLUMNS = ('state', 'attempts', 'last_error', 'started_at', 'finished_at', 'run_after')

TENANT_PREDICATE = 'organization_id = docpilot_current_organization_id()'
QUEUE_PREDICATE = "current_setting('docpilot.queue_scope', true) = 'on'"


def upgrade() -> None:
    """Upgrade schema."""
    # Two databases migrating at once can both find the role missing; the loser
    # of the race gets duplicate_object or unique_violation and moves on.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'docpilot_app') THEN
                CREATE ROLE docpilot_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB;
            END IF;
        EXCEPTION WHEN duplicate_object OR unique_violation THEN
            NULL;
        END
        $$
        """
    )
    op.execute('GRANT USAGE ON SCHEMA public TO docpilot_app')
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON {", ".join(APP_TABLES)} TO docpilot_app')
    # jobs is the one table the application updates by column. The queue policy
    # admits every row to an UPDATE, and a policy cannot see the row's old values,
    # so nothing in it stops a queue session from moving a job to another
    # organization (organization_id, document_id, ask_run_id). Only the columns
    # the worker changes are granted; the rest are fixed once the job is inserted.
    op.execute('REVOKE UPDATE ON jobs FROM docpilot_app')
    op.execute(f'GRANT UPDATE ({", ".join(WORKER_UPDATED_JOB_COLUMNS)}) ON jobs TO docpilot_app')

    for table in CONTENT_TABLES:
        op.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY')
        op.execute(
            f'CREATE POLICY tenant_isolation ON {table} FOR ALL '
            f'USING ({TENANT_PREDICATE}) WITH CHECK ({TENANT_PREDICATE})'
        )

    # Permissive policies are ORed, so these only add to tenant_isolation. The
    # claim's SELECT ... FOR UPDATE needs both the SELECT and the UPDATE policy.
    op.execute(f'CREATE POLICY jobs_queue_claim ON jobs FOR SELECT USING ({QUEUE_PREDICATE})')
    op.execute(f'CREATE POLICY jobs_queue_update ON jobs FOR UPDATE USING ({QUEUE_PREDICATE})')

    # The same four sums as app.budget.spent_today_usd, over every organization.
    op.execute(
        """
        CREATE FUNCTION docpilot_spend_since(since timestamptz) RETURNS numeric
        LANGUAGE sql STABLE SECURITY DEFINER
        SET search_path = pg_catalog, public, pg_temp
        AS $$
            SELECT coalesce((SELECT sum(cost_usd) FROM extractions WHERE created_at >= since), 0)
                 + coalesce((SELECT sum(cost_usd) FROM document_pages
                             WHERE source = 'transcription' AND created_at >= since), 0)
                 + coalesce((SELECT sum(cost_usd) FROM ask_runs WHERE created_at >= since), 0)
                 + coalesce((SELECT sum(judge_cost_usd) FROM ask_runs WHERE judged_at >= since), 0)
        $$
        """
    )
    op.execute('REVOKE ALL ON FUNCTION docpilot_spend_since(timestamptz) FROM PUBLIC')
    op.execute('GRANT EXECUTE ON FUNCTION docpilot_spend_since(timestamptz) TO docpilot_app')


def downgrade() -> None:
    """Downgrade schema."""
    op.execute('DROP FUNCTION docpilot_spend_since(timestamptz)')
    op.execute('DROP POLICY jobs_queue_update ON jobs')
    op.execute('DROP POLICY jobs_queue_claim ON jobs')
    for table in reversed(CONTENT_TABLES):
        op.execute(f'DROP POLICY tenant_isolation ON {table}')
        op.execute(f'ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE {table} DISABLE ROW LEVEL SECURITY')

    # Guarded: the role may already be gone (dropped by hand, or by the
    # downgrade of another database that held the last grant). REVOKE ALL on a
    # table also revokes its column privileges (the columns of jobs).
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'docpilot_app') THEN
                REVOKE ALL ON {", ".join(APP_TABLES)} FROM docpilot_app;
                REVOKE ALL ON SCHEMA public FROM docpilot_app;
                IF EXISTS (
                    SELECT 1 FROM pg_auth_members m JOIN pg_roles g ON g.oid = m.roleid
                    WHERE g.rolname = 'docpilot_app'
                ) THEN
                    RAISE NOTICE 'role docpilot_app kept: login roles are still members of it';
                ELSE
                    BEGIN
                        DROP ROLE docpilot_app;
                    EXCEPTION WHEN dependent_objects_still_exist THEN
                        RAISE NOTICE 'role docpilot_app kept: another database on this cluster still grants to it';
                    END;
                END IF;
            END IF;
        END
        $$
        """
    )
