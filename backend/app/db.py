from collections.abc import AsyncGenerator
from functools import partial

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.auth.principal import Principal, current_principal
from app.config import get_settings
from app.models import IDENTITY_TABLES, Base
from app.tenancy import SessionFactory, bind_tenant, tenant_session

settings = get_settings()

engine = create_async_engine(settings.database_url)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_session(
    principal: Principal = Depends(current_principal),
) -> AsyncGenerator[AsyncSession]:
    """A session bound to the caller's organization: every query it runs is
    scoped to it, and a route that touches data cannot run without one."""
    async with async_session_maker() as session:
        await bind_tenant(session, principal.organization_id)
        yield session


def get_tenant_session_factory(
    principal: Principal = Depends(current_principal),
) -> SessionFactory:
    """For work that outlives its request -- a streamed /ask run keeps
    going after its client disconnects, so it can't borrow the request's
    session, which FastAPI closes when the response ends. Each session the
    factory opens is bound to the caller's organization. A dependency so
    tests can hand it their own transaction."""
    return partial(tenant_session, async_session_maker, principal.organization_id)


class TenantIsolationError(RuntimeError):
    """The database role this process connects as would skip row-level
    security, or a content table has none."""


# What the role of this connection is and can become, and what the database
# holds. Role attributes (superuser, BYPASSRLS) are not inherited, but a member
# of such a role, or of the documents table's owner, can SET ROLE into it
# whether or not the membership inherits (pg_has_role 'MEMBER' ignores that), so
# every other role the connection could become is checked too. A superuser is a
# member of every role, so can_become is only asked of the rest.
_ISOLATION_SQL = text(
    """
    WITH documents_owner AS (
        SELECT c.relowner AS oid FROM pg_class c WHERE c.oid = to_regclass('public.documents')
    )
    SELECT r.rolname,
           r.rolsuper,
           r.rolbypassrls,
           EXISTS (SELECT 1 FROM documents_owner o WHERE o.oid = r.oid) AS owns_documents,
           NOT r.rolsuper AND EXISTS (
               SELECT 1 FROM pg_roles g
               WHERE g.oid <> r.oid
                 AND (g.rolsuper OR g.rolbypassrls OR g.oid IN (SELECT oid FROM documents_owner))
                 AND pg_has_role(r.oid, g.oid, 'MEMBER')
           ) AS can_become_a_bypassing_role,
           ARRAY(
               SELECT t FROM unnest(CAST(:content_tables AS text[])) AS t
               WHERE to_regclass('public.' || quote_ident(t)) IS NULL
               ORDER BY t
           ) AS missing_tables,
           ARRAY(
               SELECT c.relname FROM pg_class c
               WHERE c.relkind = 'r' AND c.relnamespace = 'public'::regnamespace
                 AND c.relname = ANY(:content_tables) AND NOT c.relrowsecurity
               ORDER BY c.relname
           ) AS tables_without_rls
    FROM pg_roles r
    WHERE r.rolname = current_user
    """
)


async def verify_tenant_isolation(engine: AsyncEngine) -> None:
    """Refuse to serve unless row-level security binds this process: its role
    must not be a superuser, must not have BYPASSRLS, must not own the
    documents table, and must not be able to become (SET ROLE, through any
    membership) a role that does any of those; every content table must exist
    and have row-level security enabled. A host backend/.env that still names the
    database owner would otherwise turn the second tenancy layer off without a
    sound, and a database with no tables would pass for want of anything to
    check. There is no setting that skips this check.

    Called by the API's lifespan, the worker's main, and the stdio MCP server.
    Raises TenantIsolationError."""
    content_tables = sorted(name for name in Base.metadata.tables if name not in IDENTITY_TABLES)
    async with engine.connect() as connection:
        role, superuser, bypass_rls, owns_documents, can_become, missing, without_rls = (
            await connection.execute(_ISOLATION_SQL, {"content_tables": content_tables})
        ).one()
    problems = []
    if superuser:
        problems.append("is a superuser")
    if bypass_rls:
        problems.append("has BYPASSRLS")
    if owns_documents:
        problems.append("owns the documents table")
    if can_become:
        problems.append("can become a role that skips row-level security")
    if missing:
        problems.append(f"connects to a database without {', '.join(missing)}")
    if without_rls:
        problems.append(f"connects to a database where {', '.join(without_rls)} has no row-level security")
    if problems:
        raise TenantIsolationError(
            f"database role {role!r} {' and '.join(problems)}, so tenant isolation would not hold. "
            "Connect as the application role: DATABASE_URL is the runtime role "
            "(scripts/provision_db_role.py creates it), MIGRATION_DATABASE_URL the owner "
            "(alembic upgrade head)."
        )
