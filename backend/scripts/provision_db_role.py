"""Create or update the application's login role (app/db_roles.py).

Run after `alembic upgrade head`, as the database owner:

    python scripts/provision_db_role.py

MIGRATION_DATABASE_URL (the owner) is the connection that creates the role;
DATABASE_URL names the role to create, its user and its password. The compose
`migrate` service and CI run it with both set. It refuses a DATABASE_URL that
names the owner, or a different server or database than MIGRATION_DATABASE_URL,
and it never prints a password.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db_roles import GROUP_ROLE, ProvisioningError, provision_runtime_role


async def main() -> int:
    settings = get_settings()
    if not settings.migration_database_url:
        print("error: MIGRATION_DATABASE_URL (the database owner) is not set", file=sys.stderr)
        return 1
    try:
        role = await provision_runtime_role(settings.migration_database_url, settings.database_url)
    except ProvisioningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"provisioned login role {role} (member of {GROUP_ROLE})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
