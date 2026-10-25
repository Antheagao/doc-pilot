from collections.abc import AsyncGenerator
from functools import partial

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth.principal import Principal, current_principal
from app.config import get_settings
from app.models import Base  # noqa: F401  (declared in models.py; imported from here throughout)
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
