"""Row-level tenancy in the application: every query runs for one organization.

A session is bound to an organization (`bind_tenant`, or `tenant_session`, which
does it for a block). Binding does two things, both fail-closed.

1. At the start of every transaction the session applies the organization as
   the transaction-local setting `docpilot.organization_id`. The column default
   of every content table reads it (docpilot_current_organization_id()), which
   is how a new row gets its organization without code ever setting one. It
   must be applied to each transaction, not once: a commit or rollback ends
   the setting, so a listener on `after_begin` re-applies it, including for the
   transaction a handler's rollback() begins, and for the savepoint sessions the
   test harness uses.
2. Every ORM select, update, and delete is filtered to the organization
   (`with_loader_criteria` on TenantMixin, aliases included), so a query that
   forgets its own filter still cannot see another tenant's rows.

A session with no organization refuses instead of guessing: a statement that
touches a content table, or a flush of a content row, raises TenantScopeError
before any SQL is sent. The one exception is `queue_session`, for the worker's
claim of the oldest pending job in any organization: it may touch only the
models marked `__queue_scoped__` (the job queue), and the claim then binds the
job's organization before it reads anything else.

Every kind of statement the ORM hook can see is either filtered or refused:
selects (including UNIONs), updates and deletes get the filter; a from_statement
over a content table, a statement built from Table objects, a bulk update or
delete by primary key, an INSERT ... SELECT, and an insert or update that names
organization_id are refused, because the filter cannot be applied to them or
they would put a row in the wrong organization.

What is left is what the ORM hook cannot filter: raw `text()` and SQL sent
straight to a connection (`session.connection().execute(...)`), a Core Table
nested inside an otherwise ORM statement, a `session.get` of an object already
in the session's identity map (no SQL), and a refresh of one (the filter does
not apply to refreshes; `bind_tenant` empties the identity map when it
switches organizations for that reason). The application issues none of these
against content tables. A database layer (row-level security under a non-owner
role) is meant to catch them; this module is the layer that runs on every
connection, including an owner's.

`after_begin` also fires when `begin_nested` opens a savepoint, so each tool
call's savepoint (app/agent/tools.py execute_tool) sends the setting again.
That is cheap and harmless.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from sqlalchemy import (
    Delete,
    Insert,
    Select,
    Update,
    event,
    false,
    inspect,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import ORMExecuteState, Session, with_loader_criteria
from sqlalchemy.orm.context import FromStatement
from sqlalchemy.sql import visitors
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import CompoundSelect

from app.models import Organization, TenantMixin

# The keys of Session.info that hold the binding.
TENANT_KEY = "docpilot.organization_id"
QUEUE_KEY = "docpilot.queue_scope"

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class TenantScopeError(RuntimeError):
    """A session touched tenant data without an organization bound to it (or,
    for a queue session, touched anything but the job queue)."""


def _tenant_classes() -> list[type]:
    """Every mapped class that carries an organization."""
    return list(TenantMixin.__subclasses__())


def _queue_classes() -> list[type]:
    return [cls for cls in _tenant_classes() if getattr(cls, "__queue_scoped__", False)]


def _tables_of(statement: Any) -> set[str]:
    """The names of the tables a statement reads or writes, however deep: its
    FROM list and joins, subqueries, CTEs, the branches of a UNION, and the
    target of an insert, update, or delete. Names, not objects, so an alias of
    a table counts as the table."""
    names: set[str] = set()
    pending = [statement]
    seen: set[int] = set()
    while pending:
        element = pending.pop()
        for node in visitors.iterate(element):
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, Table):
                names.add(node.name)
            elif isinstance(node, Select):
                pending.extend(node.get_final_froms())
            elif isinstance(node, Insert | Update | Delete):
                pending.append(node.table)
    return names


def _sets_organization(statement: Any, parameters: Any) -> bool:
    """Whether an insert or update names organization_id: in .values(), in a
    multi-row .values(), or in the parameter dicts of a bulk execution. A row's
    organization is the transaction's, never a value some code supplies."""

    def named(keys: Any) -> bool:
        return any(getattr(key, "key", key) == "organization_id" for key in keys)

    if named(getattr(statement, "_values", None) or {}):
        return True
    if named(column for column, _ in getattr(statement, "_ordered_values", None) or ()):
        return True
    for rows in getattr(statement, "_multi_values", ()):
        if any(isinstance(row, dict) and named(row) for row in rows):
            return True
    rows = parameters if isinstance(parameters, list | tuple) else [parameters]
    return any(isinstance(row, dict) and named(row) for row in rows)


def _refuse(message: str) -> TenantScopeError:
    return TenantScopeError(
        f"{message}: bind an organization first (app.tenancy.tenant_session or bind_tenant)"
    )


@event.listens_for(Session, "after_begin")
def _apply_tenant(session: Session, transaction: Any, connection: Any) -> None:
    organization_id = session.info.get(TENANT_KEY)
    if organization_id is not None:
        connection.execute(
            text("SELECT set_config('docpilot.organization_id', :organization_id, true)"),
            {"organization_id": str(organization_id)},
        )


_SCOPED_STATEMENTS = (Select, CompoundSelect, FromStatement, Update, Delete, Insert)


@event.listens_for(Session, "do_orm_execute")
def _scope_statement(state: ORMExecuteState) -> None:
    statement = state.statement
    if not isinstance(statement, _SCOPED_STATEMENTS):
        return
    session = state.session
    tenant_tables = {cls.__table__.name for cls in _tenant_classes()}
    # The mappers cover a from_statement(text(...)), whose SQL has no tables
    # the scan could find.
    tables = (_tables_of(statement) | {mapper.local_table.name for mapper in state.all_mappers}) & tenant_tables
    if tables and isinstance(statement, FromStatement):
        raise TenantScopeError(
            f"a from_statement over {sorted(tables)} cannot be filtered to an organization; "
            "write the query as a select"
        )
    organization_id = session.info.get(TENANT_KEY)
    if organization_id is None:
        # No organization: refuse a statement on a table this session may not
        # touch. Should the scan of the statement ever miss a form, the same
        # classes are also filtered to no rows, so the miss reads nothing.
        _require_scope(session, tables)
        if not isinstance(statement, Insert | FromStatement):
            state.statement = statement.options(
                *(with_loader_criteria(cls, false(), include_aliases=True) for cls in _blocked_classes(session))
            )
        return
    if tables:
        _require_filterable(state, statement, tables)
    if not isinstance(statement, Insert | FromStatement):
        state.statement = statement.options(
            with_loader_criteria(
                TenantMixin,
                lambda cls: cls.organization_id == organization_id,
                include_aliases=True,
            )
        )


def _require_filterable(state: ORMExecuteState, statement: Any, tables: set[str]) -> None:
    """For a session with an organization: refuse the forms on its content
    tables that the filter cannot be applied to, instead of letting them run
    across organizations."""
    if not state.is_orm_statement:
        raise TenantScopeError(
            f"a statement on the tables {sorted(tables)} was built from Table objects, which the "
            "organization filter does not apply to; use the mapped classes"
        )
    if isinstance(statement, Update | Delete) and state.is_executemany:
        raise TenantScopeError(
            f"a bulk update or delete of {sorted(tables)} by primary key is not filtered to the "
            "organization; update the loaded objects, or write a where clause"
        )
    if isinstance(statement, Insert) and statement.select is not None:
        raise TenantScopeError(
            f"an INSERT ... SELECT into {sorted(tables)} cannot take its rows from the "
            "organization alone; add the rows in Python"
        )
    if isinstance(statement, Insert | Update) and _sets_organization(statement, state.parameters):
        raise TenantScopeError(
            "organization_id is not a value to supply: a row takes the session's organization"
        )


def _blocked_classes(session: Session) -> list[type]:
    """The content models a session with no organization may not touch."""
    allowed = _queue_classes() if session.info.get(QUEUE_KEY) else []
    return [cls for cls in _tenant_classes() if cls not in allowed]


def _require_scope(session: Session, statement_tables: set[str]) -> None:
    """Raise TenantScopeError if a session with no organization is about to
    touch a table it may not."""
    blocked = {cls.__table__.name for cls in _blocked_classes(session)} & statement_tables
    if not blocked:
        return
    if session.info.get(QUEUE_KEY):
        allowed = sorted(cls.__table__.name for cls in _queue_classes())
        raise _refuse(f"a queue session may touch only {allowed}, not {sorted(blocked)}")
    raise _refuse(f"a statement touches {sorted(blocked)} on a session with no organization")


@event.listens_for(Session, "before_flush")
def _scope_flush(session: Session, flush_context: Any, instances: Any) -> None:
    organization_id = session.info.get(TENANT_KEY)
    queue = session.info.get(QUEUE_KEY)
    for instance in (*session.new, *session.dirty, *session.deleted):
        if not isinstance(instance, TenantMixin):
            continue
        if organization_id is not None:
            # Rows take their organization from the transaction. One that
            # already names another organization was built for it by mistake.
            if instance in session.new and instance.organization_id not in (None, organization_id):
                raise TenantScopeError(
                    f"a {type(instance).__name__} for another organization was added to a "
                    "session bound to this one"
                )
            if instance in session.dirty and inspect(instance).attrs.organization_id.history.has_changes():
                raise TenantScopeError(
                    f"the organization of a {type(instance).__name__} cannot be changed"
                )
        elif not (queue and getattr(instance, "__queue_scoped__", False)):
            raise _refuse(f"a {type(instance).__name__} is flushed on a session with no organization")


async def bind_tenant(session: AsyncSession, organization_id: uuid.UUID) -> None:
    """Bind `session` to an organization from now on. If a transaction is
    already open, the organization is applied to it at once; later transactions
    get it as they begin.

    Binding a different organization to a session that already has one flushes
    what it holds for the old one, then empties the session's identity map: the
    ORM's criteria do not apply to a refresh, so an object loaded for one
    organization must not stay reachable from another."""
    previous = session.info.get(TENANT_KEY)
    if previous is not None and previous != organization_id:
        await session.flush()
        session.expunge_all()
    session.info[TENANT_KEY] = organization_id
    if session.in_transaction():
        await session.execute(
            text("SELECT set_config('docpilot.organization_id', :organization_id, true)"),
            {"organization_id": str(organization_id)},
        )


@asynccontextmanager
async def tenant_session(
    factory: Callable[[], AbstractAsyncContextManager[AsyncSession]], organization_id: uuid.UUID
) -> AsyncIterator[AsyncSession]:
    """A session from `factory`, bound to `organization_id` for the block."""
    async with factory() as session:
        await bind_tenant(session, organization_id)
        yield session


@asynccontextmanager
async def queue_session(
    factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> AsyncIterator[AsyncSession]:
    """A session for the worker's claim: it may read and update the job queue
    across organizations and nothing else. The claim binds the job's
    organization (`bind_tenant`) before it touches a document."""
    async with factory() as session:
        session.info[QUEUE_KEY] = True
        yield session


async def ephemeral_organization(session: AsyncSession) -> Organization:
    """A new organization for a run that must not mix with real data (an eval
    corpus), created in the caller's transaction and bound to the session. The
    callers roll the transaction back, so nothing of it is left behind."""
    organization = Organization(
        name="Ephemeral run", slug=f"ephemeral-{uuid.uuid4().hex[:12]}"
    )
    session.add(organization)
    await session.flush()
    await bind_tenant(session, organization.id)
    return organization


async def organization_ids(session: AsyncSession, slug: str | None = None) -> list[uuid.UUID]:
    """Every organization's id, oldest first: what a maintenance script
    iterates to act on each organization in turn. With `slug`, only that
    organization's; raises ValueError when there is none."""
    stmt = select(Organization.id).order_by(Organization.created_at, Organization.id)
    if slug is not None:
        stmt = stmt.where(Organization.slug == slug)
    ids = list((await session.execute(stmt)).scalars())
    if slug is not None and not ids:
        raise ValueError(f"no organization with slug {slug!r}")
    return ids
