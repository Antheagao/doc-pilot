"""Create and manage organizations, users, memberships, and API keys.

There is no signup route: accounts exist because an operator made them here.
Run from backend/ with the venv active, or in the stack with
`docker compose exec api python scripts/accounts.py ...` (add `-T` when a
password comes in on stdin):

    python scripts/accounts.py create-organization --name "Acme" --slug acme
    python scripts/accounts.py create-user --email ada@example.com
    python scripts/accounts.py add-member --organization acme --email ada@example.com --role owner
    python scripts/accounts.py set-password --email ada@example.com
    python scripts/accounts.py create-api-key --organization acme --name "laptop"
    python scripts/accounts.py prune-sessions

A password is never an argument (it would sit in the shell history and the
process list): it is prompted for twice, or with --password-stdin read as the
first line of stdin. `create-api-key` prints the key alone on stdout, once; it
is not stored and cannot be shown again. set-password ends the user's sessions.
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import accounts, api_keys, sessions
from app.db import async_session_maker, engine
from app.models import MEMBERSHIP_ROLE_MEMBER, MEMBERSHIP_ROLES


def read_password(*, from_stdin: bool) -> str:
    """The password for a new or changed account: the first line of stdin, or
    prompted for twice. Raises ValueError when the two entries differ or stdin
    is empty."""
    if from_stdin:
        line = sys.stdin.readline()
        if not line:
            raise ValueError("--password-stdin needs the password on the first line of stdin")
        return line.rstrip("\r\n")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Password (again): ") != password:
        raise ValueError("the passwords do not match")
    return password


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser whose error never repeats what was typed: argparse
    echoes unrecognized arguments, which would print a password that was passed
    as `--password hunter2` into the terminal and its logs."""

    def error(self, message: str):
        if message.startswith("unrecognized arguments:"):
            message = (
                "unrecognized arguments (not shown); a password is never an argument, "
                "use the prompt or --password-stdin"
            )
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def refusal_message(exc: IntegrityError) -> str:
    """One line for a constraint the database refused. Only the constraint is
    named: the exception's own text carries the statement and its parameters (a
    password hash, a key digest), which must not reach a terminal or a log."""
    constraint = getattr(exc.orig, "constraint_name", None) or getattr(
        exc.orig.__cause__, "constraint_name", None
    )
    return "the database refused the change" + (f" ({constraint})" if constraint else "")


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: `--password` must not quietly mean `--password-stdin`, and
    # must not read as an option that takes the password.
    parser = _Parser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    create_organization = commands.add_parser(
        "create-organization", allow_abbrev=False, help="create an organization"
    )
    create_organization.add_argument("--name", required=True)
    create_organization.add_argument("--slug", required=True)

    create_user = commands.add_parser("create-user", allow_abbrev=False, help="create a user")
    create_user.add_argument("--email", required=True)
    add_member = commands.add_parser(
        "add-member", allow_abbrev=False, help="add a user to an organization"
    )
    add_member.add_argument("--organization", required=True, metavar="SLUG")
    add_member.add_argument("--email", required=True)
    add_member.add_argument("--role", choices=MEMBERSHIP_ROLES, default=MEMBERSHIP_ROLE_MEMBER)
    set_password = commands.add_parser(
        "set-password", allow_abbrev=False, help="replace a user's password and end their sessions"
    )
    set_password.add_argument("--email", required=True)
    for command in (create_user, set_password):
        command.add_argument(
            "--password-stdin",
            action="store_true",
            help="read the password from the first line of stdin instead of prompting",
        )

    create_api_key = commands.add_parser(
        "create-api-key", allow_abbrev=False, help="create an API key and print it once"
    )
    create_api_key.add_argument("--organization", required=True, metavar="SLUG")
    create_api_key.add_argument("--name", required=True)

    commands.add_parser("prune-sessions", allow_abbrev=False, help="delete expired sessions")
    return parser


async def run(session: AsyncSession, args: argparse.Namespace) -> int:
    """Run one command in the caller's session and commit it. Raises ValueError
    (AccountError included) for input the operator can fix; nothing is committed
    then."""
    api_key_token = None
    match args.command:
        case "create-organization":
            organization = await accounts.create_organization(
                session, name=args.name, slug=args.slug
            )
            print(f"created organization {organization.slug} ({organization.id})")
        case "create-user":
            user = await accounts.create_user(
                session,
                email=args.email,
                password=read_password(from_stdin=args.password_stdin),
            )
            print(f"created user {user.email} ({user.id})")
        case "add-member":
            organization = await accounts.get_organization(session, args.organization)
            user = await accounts.get_user(session, args.email)
            await accounts.add_membership(
                session, organization_id=organization.id, user_id=user.id, role=args.role
            )
            print(f"added {user.email} to {organization.slug} as {args.role}")
        case "set-password":
            user = await accounts.get_user(session, args.email)
            ended = await accounts.set_password(
                session, user, read_password(from_stdin=args.password_stdin)
            )
            print(f"changed the password of {user.email}; ended {ended} session(s)")
        case "create-api-key":
            organization = await accounts.get_organization(session, args.organization)
            key, token = await api_keys.create_api_key(
                session, organization_id=organization.id, name=args.name
            )
            api_key_token = token
            print(
                f"created API key {key.prefix} for {organization.slug}; "
                "it is shown once and cannot be recovered",
                file=sys.stderr,
            )
        case "prune-sessions":
            removed = await sessions.delete_expired_sessions(session)
            print(f"deleted {removed} expired session(s)")
    await session.commit()
    if api_key_token is not None:
        # The one secret this script prints, only once it is stored, and alone
        # on stdout so `$(...)` captures exactly it.
        print(api_key_token)
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        async with async_session_maker() as session:
            return await run(session, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except IntegrityError as exc:
        print(f"error: {refusal_message(exc)}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
