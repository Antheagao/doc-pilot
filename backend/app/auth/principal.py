"""Who is making a request, as far as the data layer needs to know.

A placeholder until sign-in exists: every request is the default organization,
which is what the app was before it had tenants. The routes and the MCP server
already ask for a principal and bind its organization to their session, so
replacing `current_principal` is the only change sign-in has to make to them.
"""

import uuid
from dataclasses import dataclass

from app.models import DEFAULT_ORGANIZATION_ID


@dataclass(frozen=True)
class Principal:
    """The caller: for now just the organization whose data it may see."""

    organization_id: uuid.UUID


async def current_principal() -> Principal:
    """The principal of the current request (a FastAPI dependency): always the
    default organization."""
    return Principal(organization_id=DEFAULT_ORGANIZATION_ID)
