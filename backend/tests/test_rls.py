"""Row-level security (migration 73a7d8691932), the database layer of tenancy,
shown on its own: every statement here is raw `text()` SQL on the suite's
non-owner role, which the ORM layer of app/tenancy.py never sees, so what holds
is the database's doing. (tests/test_tenancy.py shows the ORM layer alone, on an
owner connection.)

The tenant is the transaction-local setting `docpilot.organization_id`. The
tests set it with set_config() themselves, inside a savepoint when a statement
is meant to fail: opening a savepoint applies the session's bound organization
again (app/tenancy.py), so the setting under test is made after it opens.
"""

import asyncio
import base64
import os
import subprocess
import sys
import traceback
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.budget import day_start, installation_spent_today_usd, spent_today_usd
from app.config import BACKEND_DIR
from app.db import TenantIsolationError, engine, verify_tenant_isolation
from app.db_roles import (
    GROUP_ROLE,
    ProvisioningError,
    check_urls,
    ensure_runtime_role,
    provision_runtime_role,
    scram_verifier,
)
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    IDENTITY_TABLES,
    AskRun,
    Base,
    Document,
    DocumentPage,
    Extraction,
    Job,
    Organization,
)
from app.tenancy import bind_tenant, queue_session

CONTENT_TABLES = [
    "documents",
    "jobs",
    "extractions",
    "extracted_fields",
    "reconciliations",
    "document_pages",
    "document_chunks",
    "ask_runs",
]

# A column an UPDATE can rewrite to itself, for statements that must reach a
# row and change nothing. The application may update jobs by column only.
NOOP_COLUMN = {table: "organization_id" for table in CONTENT_TABLES} | {"jobs": "state"}

# (columns, values) of a valid row, every table's required columns. The
# parents are bound parameters: :doc, :ext, :page, :run (one of the target
# organization's own rows), and :org is the organization_id the row names.
INSERT_ROWS = {
    "documents": ("id, filename, mime_type, storage_path, status", "gen_random_uuid(), 'f', 'm', 'p', 'uploaded'"),
    "jobs": ("id, document_id, state, attempts", "gen_random_uuid(), :doc, 'pending', 0"),
    "extractions": (
        "id, document_id, prompt_version, model, raw_response, input_tokens, output_tokens, cost_usd, latency_ms",
        "gen_random_uuid(), :doc, 'p', 'm', '{}', 1, 1, 0, 1",
    ),
    "extracted_fields": (
        "id, extraction_id, field_name, value, confidence, needs_review",
        "gen_random_uuid(), :ext, 'f' || gen_random_uuid(), '{}', 0.9, false",
    ),
    "reconciliations": (
        "id, extraction_id, rules_version, status, report",
        "gen_random_uuid(), :ext, 'v', 'pass', '{}'",
    ),
    "document_pages": (
        "id, document_id, page_number, text, source, input_tokens, output_tokens, cost_usd, latency_ms",
        "gen_random_uuid(), :doc, (random() * 1e9)::int + 10, 't', 'gold', 0, 0, 0, 0",
    ),
    "document_chunks": (
        (
            "id, document_id, page_id, page_number, chunk_index, text, char_start, char_end, embedding, "
            "embedding_model"
        ),
        (
            "gen_random_uuid(), :doc, :page, 1, (random() * 1e9)::int + 10, 't', 0, 1, "
            "array_fill(0::real, ARRAY[384])::vector, 'm'"
        ),
    ),
    "ask_runs": (
        (
            "id, question, status, answer, citations, tool_calls, evidence, steps, model, prompt_version, "
            "input_tokens, output_tokens, cache_read_input_tokens, cost_usd, latency_ms, unresolved_citations"
        ),
        "gen_random_uuid(), 'q', 'answered', 'a', '[]', '[]', '', 1, 'm', 'v', 0, 0, 0, 0, 1, 0",
    ),
}


def _insert(table: str, *, organization: bool = True) -> str:
    columns, values = INSERT_ROWS[table]
    if organization:
        return f"INSERT INTO {table} ({columns}, organization_id) VALUES ({values}, :org)"
    return f"INSERT INTO {table} ({columns}) VALUES ({values})"


def _parents(ids: dict[type, uuid.UUID]) -> dict[str, uuid.UUID]:
    return {"doc": ids[Document], "ext": ids[Extraction], "page": ids[DocumentPage], "run": ids[AskRun]}


async def _set_tenant(session: AsyncSession, organization_id: uuid.UUID | None) -> None:
    """Set the tenant the way the app does, but by hand, with nothing of the
    ORM layer in between. None leaves it unset."""
    await session.execute(
        text("SELECT set_config('docpilot.organization_id', :organization_id, true)"),
        {"organization_id": "" if organization_id is None else str(organization_id)},
    )


async def _count(session: AsyncSession, table: str, organization_id: uuid.UUID | None = None) -> int:
    where = "" if organization_id is None else " WHERE organization_id = :organization_id"
    return (
        await session.execute(
            text(f"SELECT count(*) FROM {table}{where}"), {"organization_id": organization_id}
        )
    ).scalar_one()


# --- the tenant_isolation policy ---------------------------------------------


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_count_sees_only_the_tenants_rows_and_none_without_one(
    db_session: AsyncSession, second_organization: Organization, two_tenants, table: str
) -> None:
    ours, theirs = DEFAULT_ORGANIZATION_ID, second_organization.id

    await _set_tenant(db_session, ours)
    assert await _count(db_session, table) == await _count(db_session, table, ours) >= 1
    assert await _count(db_session, table, theirs) == 0

    await _set_tenant(db_session, theirs)
    assert await _count(db_session, table) == await _count(db_session, table, theirs) >= 1
    assert await _count(db_session, table, ours) == 0

    await _set_tenant(db_session, None)
    assert await _count(db_session, table) == 0


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_insert_without_a_tenant_is_refused(
    db_session: AsyncSession, two_tenants, table: str
) -> None:
    ours, _ = two_tenants

    # Naming an organization is no help without a tenant, and neither is
    # leaving the column to its default, which is NULL without one.
    for sql in (_insert(table), _insert(table, organization=False)):
        with pytest.raises(DBAPIError, match="row-level security"):
            async with db_session.begin_nested():
                await _set_tenant(db_session, None)
                await db_session.execute(text(sql), {**_parents(ours), "org": DEFAULT_ORGANIZATION_ID})


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_insert_naming_another_organization_is_refused(
    db_session: AsyncSession, second_organization: Organization, two_tenants, table: str
) -> None:
    _, theirs = two_tenants

    # The parents are the other organization's own, so only the policy stands
    # in the way: the row would otherwise be valid.
    with pytest.raises(DBAPIError, match="row-level security"):
        async with db_session.begin_nested():
            await _set_tenant(db_session, DEFAULT_ORGANIZATION_ID)
            await db_session.execute(text(_insert(table)), {**_parents(theirs), "org": second_organization.id})


async def _insert_for_the_tenant(session: AsyncSession, table: str, parents: dict, *, organization: bool) -> None:
    parents = dict(parents)
    if table == "reconciliations":  # one per extraction: give it a new one
        parents["ext"] = (
            await session.execute(text(_insert("extractions") + " RETURNING id"), {**parents, "org": DEFAULT_ORGANIZATION_ID})
        ).scalar_one()
    await session.execute(text(_insert(table, organization=organization)), {**parents, "org": DEFAULT_ORGANIZATION_ID})


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_insert_for_the_tenant_is_accepted_and_takes_it_by_default(
    db_session: AsyncSession, two_tenants, table: str
) -> None:
    ours, _ = two_tenants
    await _set_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    before = await _count(db_session, table, DEFAULT_ORGANIZATION_ID)

    # The positive control for the refusals above: the same row, under its
    # own tenant, with the column named and with it left to its default.
    await _insert_for_the_tenant(db_session, table, _parents(ours), organization=True)
    await _insert_for_the_tenant(db_session, table, _parents(ours), organization=False)

    after = await _count(db_session, table, DEFAULT_ORGANIZATION_ID)
    assert after - before == 2
    assert await _count(db_session, table) == after


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_update_or_delete_of_another_tenants_rows_touches_none(
    db_session: AsyncSession, second_organization: Organization, two_tenants, table: str
) -> None:
    ours, theirs = DEFAULT_ORGANIZATION_ID, second_organization.id
    await _set_tenant(db_session, ours)

    update = await db_session.execute(
        text(f"UPDATE {table} SET {NOOP_COLUMN[table]} = {NOOP_COLUMN[table]} WHERE organization_id = :other"),
        {"other": theirs},
    )
    delete = await db_session.execute(text(f"DELETE FROM {table} WHERE organization_id = :other"), {"other": theirs})
    mine = await db_session.execute(
        text(f"UPDATE {table} SET {NOOP_COLUMN[table]} = {NOOP_COLUMN[table]} WHERE organization_id = :ours"),
        {"ours": ours},
    )

    assert (update.rowcount, delete.rowcount) == (0, 0)
    assert mine.rowcount >= 1  # the same statement reaches the tenant's own rows
    await _set_tenant(db_session, theirs)
    assert await _count(db_session, table, theirs) >= 1


@pytest.mark.parametrize("table", CONTENT_TABLES)
async def test_a_raw_update_cannot_hand_a_row_to_another_organization(
    db_session: AsyncSession, second_organization: Organization, two_tenants, table: str
) -> None:
    # jobs is not updatable by organization_id at all (a column grant); the
    # others are refused by the policy's WITH CHECK.
    refusal = "permission denied" if table == "jobs" else "row-level security"
    with pytest.raises(DBAPIError, match=refusal):
        async with db_session.begin_nested():
            await _set_tenant(db_session, DEFAULT_ORGANIZATION_ID)
            await db_session.execute(
                text(f"UPDATE {table} SET organization_id = :other WHERE organization_id = :ours"),
                {"other": second_organization.id, "ours": DEFAULT_ORGANIZATION_ID},
            )


async def test_a_composite_foreign_key_still_refuses_a_child_of_another_tenants_parent(
    db_session: AsyncSession, two_tenants
) -> None:
    _, theirs = two_tenants

    # The row is the tenant's own and the policy admits it; the parent is
    # another organization's, which the key (checked past the policy, as the
    # table's owner) refuses.
    with pytest.raises(DBAPIError, match="foreign key"):
        async with db_session.begin_nested():
            await _set_tenant(db_session, DEFAULT_ORGANIZATION_ID)
            await db_session.execute(
                text(_insert("extractions")), {"doc": theirs[Document], "org": DEFAULT_ORGANIZATION_ID}
            )


# --- the queue scope ---------------------------------------------------------


@asynccontextmanager
async def _on_the_tests_transaction(db_session: AsyncSession) -> AsyncGenerator[AsyncSession]:
    """A second session on db_session's connection and transaction, as a
    worker's queue session is a session of its own. It is rolled back, not
    committed: a setting made in a savepoint that is released stays for the
    rest of the outer transaction (the harness's one difference from the app,
    where a commit ends the transaction), and a rollback takes it back."""
    session = AsyncSession(
        bind=await db_session.connection(), expire_on_commit=False, join_transaction_mode="create_savepoint"
    )
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()


async def test_the_queue_scope_admits_every_organizations_jobs_and_no_other_table(
    db_session: AsyncSession, second_organization: Organization, two_tenants
) -> None:
    _, theirs = two_tenants
    async with queue_session(lambda: _on_the_tests_transaction(db_session)) as session:
        await _set_tenant(session, None)  # a worker's claim has no tenant yet
        assert (
            await session.execute(text("SELECT current_setting('docpilot.queue_scope', true)"))
        ).scalar_one() == "on"

        jobs = (await session.execute(text("SELECT organization_id FROM jobs"))).scalars().all()
        assert {DEFAULT_ORGANIZATION_ID, second_organization.id} <= set(jobs)
        for table in CONTENT_TABLES:
            if table != "jobs":
                assert await _count(session, table) == 0, table

        # The claim: FOR UPDATE SKIP LOCKED, then the state change.
        claimed = (
            await session.execute(
                text("SELECT id FROM jobs WHERE id = :id FOR UPDATE SKIP LOCKED"), {"id": theirs[Job]}
            )
        ).scalar_one()
        updated = await session.execute(
            text("UPDATE jobs SET state = 'processing' WHERE id = :id"), {"id": claimed}
        )
        assert updated.rowcount == 1

        # It reads and updates the queue and does nothing else to it.
        with pytest.raises(DBAPIError, match="row-level security"):
            async with session.begin_nested():
                await session.execute(
                    text(_insert("jobs")), {**_parents(theirs), "org": second_organization.id}
                )
        assert (await session.execute(text("DELETE FROM jobs"))).rowcount == 0
        assert (
            await session.execute(text("UPDATE documents SET status = status"))
        ).rowcount == 0


@pytest.mark.parametrize(
    "assignment",
    [
        "organization_id = :org, document_id = :doc",
        "organization_id = :org",
        "document_id = :doc",
        "ask_run_id = :run",
        "kind = 'index'",
        "traceparent = 'x'",
    ],
)
async def test_a_claim_cannot_move_a_job_to_another_organization_or_rewrite_what_it_is_about(
    db_session: AsyncSession, second_organization: Organization, two_tenants, assignment: str
) -> None:
    ours, theirs = two_tenants
    # The queue policy admits every row to an UPDATE and cannot see the old
    # values, so the column grant is what stops a claim re-filing a job under
    # the other organization (the composite key only needs the pair to exist).
    parameters = {"org": second_organization.id, "doc": theirs[Document], "run": theirs[AskRun], "id": ours[Job]}
    async with queue_session(lambda: _on_the_tests_transaction(db_session)) as session:
        await _set_tenant(session, None)

        with pytest.raises(DBAPIError, match="permission denied"):
            async with session.begin_nested():
                await session.execute(text(f"UPDATE jobs SET {assignment} WHERE id = :id"), parameters)

        # The columns the worker changes are still open to it.
        moved = await session.execute(text("UPDATE jobs SET state = 'processing' WHERE id = :id"), parameters)
        assert moved.rowcount == 1


async def test_binding_a_tenant_in_a_queue_session_ends_the_queue_scope(
    db_session: AsyncSession, second_organization: Organization, two_tenants
) -> None:
    scope = text("SELECT current_setting('docpilot.queue_scope', true)")
    other_jobs = text("SELECT count(*) FROM jobs WHERE organization_id = :organization_id")
    async with queue_session(lambda: _on_the_tests_transaction(db_session)) as session:
        await _set_tenant(session, None)
        assert (await session.execute(scope)).scalar_one() == "on"
        assert (await session.execute(other_jobs, {"organization_id": DEFAULT_ORGANIZATION_ID})).scalar_one() >= 1

        # What the claim does once it has the job: bind the job's organization.
        await bind_tenant(session, second_organization.id)

        assert (await session.execute(scope)).scalar_one() == ""
        assert (await session.execute(other_jobs, {"organization_id": second_organization.id})).scalar_one() >= 1
        assert (await session.execute(other_jobs, {"organization_id": DEFAULT_ORGANIZATION_ID})).scalar_one() == 0
        assert (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one() >= 1
        # Nor does a later transaction of the session bring the scope back.
        await session.rollback()
        assert (await session.execute(scope)).scalar_one() in (None, "")
        assert (await session.execute(other_jobs, {"organization_id": DEFAULT_ORGANIZATION_ID})).scalar_one() == 0


async def test_the_queue_scope_ends_with_its_transaction_and_returns_on_the_next() -> None:
    dedicated = create_async_engine(engine.url, pool_size=1, max_overflow=0)
    setting = text("SELECT current_setting('docpilot.queue_scope', true)")
    try:
        async with queue_session(lambda: AsyncSession(dedicated, expire_on_commit=False)) as session:
            assert (await session.execute(setting)).scalar_one() == "on"
            await session.commit()
            # A new transaction of the same session: set again by after_begin.
            assert (await session.execute(setting)).scalar_one() == "on"
            await session.rollback()
            assert (await session.execute(setting)).scalar_one() == "on"
        async with AsyncSession(dedicated) as plain:  # the same pooled connection
            assert (await plain.execute(setting)).scalar_one() in (None, "")
            assert (await plain.execute(text("SELECT count(*) FROM jobs"))).scalar_one() == 0
    finally:
        await dedicated.dispose()


# --- the spend function ------------------------------------------------------


async def test_the_spend_function_sums_every_organization_while_a_raw_sum_sees_one(
    db_session: AsyncSession, second_organization: Organization, two_tenants
) -> None:
    now = datetime.now(UTC)
    ours, theirs = DEFAULT_ORGANIZATION_ID, second_organization.id
    since = day_start(now)

    mine = await spent_today_usd(db_session, now)
    raw = (
        await db_session.execute(text("SELECT sum(cost_usd) FROM extractions WHERE created_at >= :since"), {"since": since})
    ).scalar_one()
    await bind_tenant(db_session, theirs)
    other = await spent_today_usd(db_session, now)
    others = 0.0
    for organization_id in (await db_session.execute(select(Organization.id))).scalars().all():
        if organization_id not in (ours, theirs):
            await bind_tenant(db_session, organization_id)
            others += await spent_today_usd(db_session, now)
    await bind_tenant(db_session, ours)

    assert mine >= 1.0 and other == pytest.approx(1.0)  # each tenant's one extraction costs a dollar
    assert float(raw) <= mine  # the raw sum is the tenant's own
    # The function is every organization's, and a number, never a row.
    assert await installation_spent_today_usd(db_session, now) == pytest.approx(mine + other + others)


async def test_the_spend_function_is_a_definer_owned_by_a_role_row_security_does_not_bind(
    db_session: AsyncSession,
) -> None:
    row = (
        await db_session.execute(
            text(
                """
                SELECT p.prosecdef, r.rolsuper OR r.rolbypassrls, p.proconfig::text, p.proacl::text
                FROM pg_proc p JOIN pg_roles r ON r.oid = p.proowner
                WHERE p.proname = 'docpilot_spend_since'
                """
            )
        )
    ).one()
    security_definer, owner_bypasses, config, acl = row

    assert security_definer and owner_bypasses
    assert "search_path=pg_catalog, public, pg_temp" in config
    assert f"{GROUP_ROLE}=X/" in acl
    assert not acl.lstrip("{").startswith("=")  # nothing is granted to PUBLIC


# --- the schema --------------------------------------------------------------


async def test_every_table_is_a_forced_policy_table_or_an_identity_table(db_session: AsyncSession) -> None:
    rows = (
        await db_session.execute(
            text(
                """
                SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
                       EXISTS (SELECT 1 FROM pg_attribute a WHERE a.attrelid = c.oid
                               AND a.attname = 'organization_id' AND NOT a.attisdropped),
                       ARRAY(SELECT polname FROM pg_policy WHERE polrelid = c.oid ORDER BY polname)
                FROM pg_class c
                WHERE c.relkind IN ('r', 'p') AND c.relnamespace = 'public'::regnamespace
                ORDER BY c.relname
                """
            )
        )
    ).all()

    tenant_tables = set()
    for name, row_security, forced, has_organization, policies in rows:
        if name == "alembic_version":
            continue
        if name in IDENTITY_TABLES:
            # Read before any tenant is known: no policy, by design.
            assert not row_security and policies == [], name
            continue
        tenant_tables.add(name)
        assert has_organization and row_security and forced, name
        expected = ["jobs_queue_claim", "jobs_queue_update", "tenant_isolation"] if name == "jobs" else ["tenant_isolation"]
        assert policies == expected, name
    # Every model that carries an organization is one of them, and nothing else is.
    assert tenant_tables == {t for t in Base.metadata.tables if t not in IDENTITY_TABLES} == set(CONTENT_TABLES)


async def test_the_policies_say_what_they_are_meant_to(db_session: AsyncSession) -> None:
    rows = (
        await db_session.execute(
            text("SELECT tablename, policyname, cmd, permissive, qual, with_check FROM pg_policies "
                 "WHERE schemaname = 'public' ORDER BY tablename, policyname")
        )
    ).all()
    by_policy = {(table, name): (cmd, permissive, qual, check) for table, name, cmd, permissive, qual, check in rows}

    for table in CONTENT_TABLES:
        cmd, permissive, qual, check = by_policy[(table, "tenant_isolation")]
        assert (cmd, permissive) == ("ALL", "PERMISSIVE")
        assert qual == check == "(organization_id = docpilot_current_organization_id())"
    claim = by_policy[("jobs", "jobs_queue_claim")]
    update = by_policy[("jobs", "jobs_queue_update")]
    assert claim[:2] == ("SELECT", "PERMISSIVE") and update[:2] == ("UPDATE", "PERMISSIVE")
    assert claim[2] == update[2] == "(current_setting('docpilot.queue_scope'::text, true) = 'on'::text)"
    assert len(rows) == len(CONTENT_TABLES) + 2


async def test_the_application_role_holds_exactly_the_grants_it_needs(db_session: AsyncSession) -> None:
    tables = (
        await db_session.execute(
            text("SELECT relname FROM pg_class WHERE relkind IN ('r', 'p') AND relnamespace = 'public'::regnamespace")
        )
    ).scalars().all()

    for table in tables:
        privileges = {
            privilege: (
                await db_session.execute(
                    text("SELECT has_table_privilege(:role, :table, :privilege)"),
                    {"role": GROUP_ROLE, "table": table, "privilege": privilege},
                )
            ).scalar_one()
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
        }
        any_update = (
            await db_session.execute(
                text("SELECT has_any_column_privilege(:role, :table, 'UPDATE')"), {"role": GROUP_ROLE, "table": table}
            )
        ).scalar_one()
        if table == "alembic_version":
            assert not any(privileges.values()) and not any_update, table
        else:
            # TRUNCATE is not subject to row-level security, so it is never granted.
            # jobs has no table-level UPDATE, only the columns the worker changes.
            assert privileges == {
                "SELECT": True, "INSERT": True, "UPDATE": table != "jobs", "DELETE": True,
                "TRUNCATE": False, "REFERENCES": False, "TRIGGER": False,
            }, table
            assert any_update, table


async def test_jobs_can_be_updated_only_in_the_columns_the_worker_changes(db_session: AsyncSession) -> None:
    granted = (
        await db_session.execute(
            text(
                """
                SELECT a.attname FROM pg_attribute a
                WHERE a.attrelid = 'jobs'::regclass AND a.attnum > 0 AND NOT a.attisdropped
                  AND has_column_privilege(:role, 'jobs', a.attname, 'UPDATE')
                ORDER BY a.attname
                """
            ),
            {"role": GROUP_ROLE},
        )
    ).scalars().all()

    assert granted == sorted(["state", "attempts", "last_error", "started_at", "finished_at", "run_after"])


# --- the role ----------------------------------------------------------------


async def test_the_suites_role_is_the_kind_row_level_security_binds(
    db_session: AsyncSession, owner_database_url: URL
) -> None:
    role, superuser, bypass, creates_roles, is_member = (
        await db_session.execute(
            text(
                "SELECT rolname, rolsuper, rolbypassrls, rolcreaterole, pg_has_role(current_user, :group, 'MEMBER') "
                "FROM pg_roles WHERE rolname = current_user"
            ),
            {"group": GROUP_ROLE},
        )
    ).one()
    owned = (
        await db_session.execute(
            text(
                """
                SELECT (SELECT count(*) FROM pg_class WHERE relowner = r.oid)
                     + (SELECT count(*) FROM pg_proc WHERE proowner = r.oid)
                     + (SELECT count(*) FROM pg_namespace WHERE nspowner = r.oid)
                     + (SELECT count(*) FROM pg_database WHERE datdba = r.oid)
                FROM pg_roles r WHERE r.rolname = current_user
                """
            )
        )
    ).scalar_one()

    assert (superuser, bypass, creates_roles, is_member) == (False, False, False, True), role
    assert owned == 0
    # It cannot become the owner, and cannot change the schema.
    with pytest.raises(DBAPIError, match="permission denied"):
        async with db_session.begin_nested():
            await db_session.execute(text(f'SET ROLE "{owner_database_url.username}"'))
    with pytest.raises(DBAPIError, match="permission denied"):
        async with db_session.begin_nested():
            await db_session.execute(text("CREATE TABLE docpilot_rls_probe (x int)"))
    with pytest.raises(DBAPIError, match="must be owner"):
        async with db_session.begin_nested():
            await db_session.execute(text("ALTER TABLE documents DISABLE ROW LEVEL SECURITY"))


# --- the startup guard -------------------------------------------------------


async def test_the_guard_accepts_the_runtime_role_and_refuses_the_owner(owner_engine: AsyncEngine) -> None:
    await verify_tenant_isolation(engine)

    with pytest.raises(TenantIsolationError, match="is a superuser") as refused:
        await verify_tenant_isolation(owner_engine)
    assert "owns the documents table" in str(refused.value)


class _FakeEngine:
    """Answers the guard's one query with a row of our choosing."""

    def __init__(self, row: tuple) -> None:
        self.row = row

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[SimpleNamespace]:
        async def execute(statement, parameters):
            return SimpleNamespace(one=lambda: self.row)

        yield SimpleNamespace(execute=execute)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("app", True, False, False, False, [], []), "is a superuser"),
        (("app", False, True, False, False, [], []), "has BYPASSRLS"),
        (("app", False, False, True, False, [], []), "owns the documents table"),
        (("app", False, False, False, True, [], []), "can become a role that skips row-level security"),
        (("app", False, False, False, False, ["jobs", "ask_runs"], []), "without jobs, ask_runs"),
        (("app", False, False, False, False, [], ["jobs", "documents"]), "jobs, documents has no row-level security"),
        (("app", True, True, True, False, [], ["jobs"]), "is a superuser and has BYPASSRLS and owns"),
    ],
)
async def test_the_guard_names_every_way_a_role_would_skip_the_policies(row: tuple, message: str) -> None:
    with pytest.raises(TenantIsolationError, match=message):
        await verify_tenant_isolation(_FakeEngine(row))


async def test_a_clean_role_passes_the_guard() -> None:
    await verify_tenant_isolation(_FakeEngine(("app", False, False, False, False, [], [])))


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class _Throwaway:
    """Roles and databases made for one test and dropped after it, never the
    suite's own: the names carry a random suffix. Roles are cluster-wide, so
    each is dropped in the order that leaves nothing behind."""

    def __init__(self, owner_url: URL) -> None:
        self.owner_url = owner_url
        self.suffix = uuid.uuid4().hex[:10]
        self.databases: list[str] = []
        self.roles: list[str] = []
        self.maintenance = create_async_engine(
            owner_url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
        )

    def name(self, kind: str) -> str:
        return f"docpilot_guard_{kind}_{self.suffix}"

    async def run(self, *statements: str) -> None:
        async with self.maintenance.connect() as connection:
            for statement in statements:
                await connection.execute(text(statement))

    async def role(self, kind: str, options: str = "", member_of: str | None = None) -> tuple[str, str]:
        name, password = self.name(kind), uuid.uuid4().hex
        await self.run(f"CREATE ROLE {_quoted(name)} LOGIN PASSWORD '{password}' {options}")
        self.roles.append(name)
        if member_of:
            await self.run(f"GRANT {_quoted(member_of)} TO {_quoted(name)}")
        return name, password

    async def database(self, kind: str) -> str:
        name = self.name(kind)
        await self.run(f"CREATE DATABASE {_quoted(name)}")
        self.databases.append(name)
        return name

    async def guard(self, role: tuple[str, str], database: str) -> None:
        name, password = role
        connecting = create_async_engine(
            self.owner_url.set(username=name, password=password, database=database), poolclass=NullPool
        )
        try:
            await verify_tenant_isolation(connecting)
        finally:
            await connecting.dispose()

    async def close(self) -> None:
        try:
            # Databases first: a role that owns a table cannot be dropped before it.
            await self.run(
                *(f"DROP DATABASE IF EXISTS {_quoted(name)} WITH (FORCE)" for name in self.databases),
                *(f"DROP ROLE IF EXISTS {_quoted(name)}" for name in reversed(self.roles)),
            )
        finally:
            await self.maintenance.dispose()


@pytest.fixture
async def throwaway(owner_database_url: URL) -> AsyncGenerator[_Throwaway]:
    made = _Throwaway(owner_database_url)
    try:
        yield made
    finally:
        await made.close()


@pytest.mark.parametrize("inherit", ["INHERIT", "NOINHERIT"])
@pytest.mark.parametrize(
    "bypassing_role_options", ["SUPERUSER", "BYPASSRLS"], ids=["a superuser", "a BYPASSRLS role"]
)
async def test_the_guard_refuses_a_role_that_can_become_a_role_that_skips_row_level_security(
    throwaway: _Throwaway, inherit: str, bypassing_role_options: str
) -> None:
    bypassing, _ = await throwaway.role("bypass", bypassing_role_options)
    member = await throwaway.role("member", inherit, member_of=bypassing)

    # Attributes are not inherited, so the role itself looks harmless; it can
    # still SET ROLE into the other.
    with pytest.raises(TenantIsolationError, match="can become a role that skips row-level security"):
        await throwaway.guard(member, throwaway.owner_url.database)


async def test_the_guard_refuses_a_role_that_is_a_member_of_the_tables_owner(throwaway: _Throwaway) -> None:
    database = await throwaway.database("owned")
    owner, _ = await throwaway.role("owner")
    # Only what the guard reads: a documents table, owned by a throwaway role.
    owned = create_async_engine(throwaway.owner_url.set(database=database), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        async with owned.connect() as connection:
            await connection.execute(text("CREATE TABLE documents (id int)"))
            await connection.execute(text(f"ALTER TABLE documents OWNER TO {_quoted(owner)}"))
    finally:
        await owned.dispose()
    member = await throwaway.role("member", "NOINHERIT", member_of=owner)
    outsider = await throwaway.role("outsider")

    with pytest.raises(TenantIsolationError, match="can become a role that skips row-level security"):
        await throwaway.guard(member, database)
    with pytest.raises(TenantIsolationError) as refused:
        await throwaway.guard(outsider, database)
    assert "can become" not in str(refused.value)  # not a member of the owner: only the empty database is wrong


async def test_the_guard_refuses_a_database_without_the_content_tables(throwaway: _Throwaway) -> None:
    database = await throwaway.database("empty")
    role = await throwaway.role("plain")

    with pytest.raises(TenantIsolationError, match="without ask_runs, document_chunks, document_pages") as refused:
        await throwaway.guard(role, database)
    assert "documents" in str(refused.value) and "jobs" in str(refused.value)
    # The same role in the migrated test database is a role the guard accepts:
    # it is not a member of anything that bypasses.
    await throwaway.guard(role, throwaway.owner_url.database)


# --- provisioning ------------------------------------------------------------


def test_the_provisioner_refuses_an_owner_runtime_url_or_another_server(owner_database_url: URL) -> None:
    owner = owner_database_url
    runtime = owner.set(username="docpilot_runtime", password="docpilot_runtime")

    assert check_urls(owner, runtime).username == "docpilot_runtime"
    with pytest.raises(ProvisioningError, match="same user"):
        check_urls(owner, owner)
    with pytest.raises(ProvisioningError, match="same user"):
        check_urls(owner, owner.set(password="another"))
    with pytest.raises(ProvisioningError, match="different servers or databases"):
        check_urls(owner, runtime.set(database="another"))
    with pytest.raises(ProvisioningError, match="different servers or databases"):
        check_urls(owner, runtime.set(port=(owner.port or 5432) + 1))
    with pytest.raises(ProvisioningError, match="different servers or databases"):
        check_urls(owner, runtime.set(host="elsewhere.invalid"))
    with pytest.raises(ProvisioningError, match="user and password"):
        check_urls(owner, runtime.set(password=""))
    with pytest.raises(ProvisioningError, match="NOLOGIN"):
        check_urls(owner, runtime.set(username=GROUP_ROLE))


async def test_provisioning_with_an_owner_runtime_url_stops_before_the_database(owner_database_url: URL) -> None:
    # No server could answer at this address: the refusal comes first.
    unreachable = owner_database_url.set(host="unreachable.invalid")

    with pytest.raises(ProvisioningError, match="same user"):
        await provision_runtime_role(unreachable, unreachable)


def test_the_provisioning_script_refuses_an_owner_runtime_url_without_printing_a_password(
    owner_database_url: URL,
) -> None:
    password = owner_database_url.password
    owner = owner_database_url.render_as_string(hide_password=False)

    result = subprocess.run(
        [sys.executable, "scripts/provision_db_role.py"],
        cwd=BACKEND_DIR,
        env={**os.environ, "MIGRATION_DATABASE_URL": owner, "DATABASE_URL": owner},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "same user" in result.stderr
    assert password not in result.stdout + result.stderr


async def test_provisioning_refuses_a_role_that_is_not_ours_and_never_prints_the_password(
    owner_engine: AsyncEngine,
) -> None:
    secret = f"pw-{uuid.uuid4().hex}"

    # docpilot_app exists and is no login role of ours: nothing is altered.
    with pytest.raises(ProvisioningError, match="refusing to alter") as refused:
        await ensure_runtime_role(owner_engine, GROUP_ROLE, secret)

    assert secret not in "".join(traceback.format_exception(refused.value))
    async with owner_engine.connect() as connection:
        can_login = (
            await connection.execute(text("SELECT rolcanlogin FROM pg_roles WHERE rolname = :role"), {"role": GROUP_ROLE})
        ).scalar_one()
    assert can_login is False


async def test_provisioning_the_suites_role_is_repeatable_and_leaves_it_able_to_log_in(
    owner_engine: AsyncEngine, owner_database_url: URL
) -> None:
    role = engine.url.username
    await ensure_runtime_role(owner_engine, role, engine.url.password)

    async with engine.connect() as connection:  # the pool's connections keep working
        assert (await connection.execute(text("SELECT current_user"))).scalar_one() == role
    fresh = create_async_engine(engine.url, poolclass=NullPool)
    try:
        async with fresh.connect() as connection:
            assert (await connection.execute(text("SELECT current_user"))).scalar_one() == role
        await verify_tenant_isolation(fresh)
    finally:
        await fresh.dispose()


# Passwords PostgreSQL's SASLprep leaves alone, rewrites (a ligature, a soft
# hyphen, a no-break space), and cannot apply to (a control character).
SCRAM_PASSWORDS = [
    "plain-ascii-" + "x" * 8,
    "p'a\"s\\s;$$ %I %L",
    "pässwörd\u2603",
    "ﬁne\u00adtuned\u00a0pass",
    "bell\u0007\u00e9",
]


@pytest.mark.parametrize("password", SCRAM_PASSWORDS)
async def test_the_scram_verifier_is_what_postgresql_itself_would_store(
    throwaway: "_Throwaway", owner_engine: AsyncEngine, password: str
) -> None:
    name = throwaway.name("scram")
    throwaway.roles.append(name)
    async with owner_engine.connect() as connection:
        connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
        await connection.execute(text("SET password_encryption = 'scram-sha-256'"))
        literal = "'" + password.replace("'", "''") + "'"  # standard_conforming_strings is on
        await connection.execute(text(f"CREATE ROLE {_quoted(name)} LOGIN PASSWORD {literal}"))
        stored = (
            await connection.execute(text("SELECT rolpassword FROM pg_authid WHERE rolname = :name"), {"name": name})
        ).scalar_one()
    prefix, salt_and_iterations, keys = stored.split("$")
    iterations, salt = salt_and_iterations.split(":")

    ours = scram_verifier(password, salt=base64.b64decode(salt), iterations=int(iterations))

    assert prefix == "SCRAM-SHA-256"
    assert ours == stored
    assert scram_verifier(password) != scram_verifier(password)  # a fresh salt each time
    assert keys.count(":") == 1


@pytest.mark.parametrize("password", SCRAM_PASSWORDS)
async def test_a_role_provisioned_with_a_verifier_logs_in_with_the_plaintext_password_alone(
    throwaway: "_Throwaway", owner_engine: AsyncEngine, owner_database_url: URL, password: str
) -> None:
    name = throwaway.name("login")
    throwaway.roles.append(name)
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(f"{statement} {parameters}")

    event.listen(owner_engine.sync_engine, "before_cursor_execute", record)
    await ensure_runtime_role(owner_engine, name, password)
    event.remove(owner_engine.sync_engine, "before_cursor_execute", record)

    # The password never left this process, in any statement or parameter.
    assert statements and not any(password in entry for entry in statements)
    assert any("SCRAM-SHA-256$" in entry for entry in statements)
    connecting = create_async_engine(
        owner_database_url.set(username=name, password=password), poolclass=NullPool
    )
    try:
        async with connecting.connect() as connection:
            assert (await connection.execute(text("SELECT current_user"))).scalar_one() == name
            assert (
                await connection.execute(text("SELECT pg_has_role(current_user, :group, 'MEMBER')"), {"group": GROUP_ROLE})
            ).scalar_one()
        await verify_tenant_isolation(connecting)
    finally:
        await connecting.dispose()
    wrong = create_async_engine(owner_database_url.set(username=name, password=password + "x"), poolclass=NullPool)
    try:
        with pytest.raises(DBAPIError):
            async with wrong.connect():
                pass
    finally:
        await wrong.dispose()


# --- the downgrade -----------------------------------------------------------


def _alembic(owner: URL, database: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    url = owner.set(database=database).render_as_string(hide_password=False)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=BACKEND_DIR,
        env={**os.environ, "MIGRATION_DATABASE_URL": url, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )


async def test_the_downgrade_keeps_a_role_another_database_still_grants_to(
    owner_engine: AsyncEngine, owner_database_url: URL
) -> None:
    """Roles are cluster-wide and grants are per database. This runs in a
    database of its own, beside the test database that always holds its
    grants (and the suite's login role, a member of it), so the downgrade must
    keep docpilot_app: either reason alone is enough for the migration, and
    here both hold. The test database itself is never downgraded."""
    name = f"docpilot_rls_downgrade_{uuid.uuid4().hex[:8]}"
    maintenance = create_async_engine(
        owner_database_url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )

    async def inspect_database() -> tuple[int, int, bool]:
        inspected = create_async_engine(owner_database_url.set(database=name), poolclass=NullPool)
        try:
            async with inspected.connect() as connection:
                policies = (await connection.execute(text("SELECT count(*) FROM pg_policies"))).scalar_one()
                forced = (
                    await connection.execute(text("SELECT count(*) FROM pg_class WHERE relforcerowsecurity"))
                ).scalar_one()
                function = (
                    await connection.execute(text("SELECT count(*) FROM pg_proc WHERE proname = 'docpilot_spend_since'"))
                ).scalar_one()
                return policies, forced, bool(function)
        finally:
            await inspected.dispose()

    async def role_exists() -> bool:
        async with owner_engine.connect() as connection:
            return bool(
                (await connection.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": GROUP_ROLE})).scalar()
            )

    async with maintenance.connect() as connection:
        await connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        up = await asyncio.to_thread(_alembic, owner_database_url, name, "upgrade", "head")
        assert up.returncode == 0, up.stderr
        assert await inspect_database() == (len(CONTENT_TABLES) + 2, len(CONTENT_TABLES), True)

        down = await asyncio.to_thread(_alembic, owner_database_url, name, "downgrade", "127aac687508")

        assert down.returncode == 0, down.stderr
        assert await inspect_database() == (0, 0, False)  # this database is back to before
        assert await role_exists()  # the test database still grants to it

        again = await asyncio.to_thread(_alembic, owner_database_url, name, "upgrade", "head")
        assert again.returncode == 0, again.stderr
        assert await inspect_database() == (len(CONTENT_TABLES) + 2, len(CONTENT_TABLES), True)
    finally:
        async with maintenance.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await maintenance.dispose()
    assert await role_exists()
