from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with async_session_maker() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """For work that outlives its request -- a streamed /ask run keeps
    going after its client disconnects, so it can't borrow the request's
    session, which FastAPI closes when the response ends. A dependency so
    tests can hand it their own transaction."""
    return async_session_maker

