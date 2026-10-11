"""Recompute the reconciliation and review routing of every extraction.

New extractions are reconciled by the worker. This backfills the rest:
extractions stored before reconciliation existed, or after a change to the
rules (`RULES_VERSION`). It makes no model call. For each extraction it runs
`refresh_reconciliation`, which stores the arithmetic report and recomputes
which fields are flagged for review; a field a person has already reviewed
is never touched. Running it twice changes nothing the second time.

Each extraction is its own transaction, so the row locks `refresh_reconciliation`
takes are held for one document at a time and never stall a reviewer, an
interrupted run keeps what it finished, and only a small summary per extraction
stays in memory. A dry run rolls each transaction back instead of committing it.

Run from backend/ with the venv active, or in the stack with
`docker compose exec api python scripts/reconcile_all.py`:

    python scripts/reconcile_all.py --dry-run   # compute, print, write nothing
    python scripts/reconcile_all.py             # compute, print, write, commit
"""

import argparse
import asyncio
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import async_session_maker, engine
from app.models import Document, Extraction
from app.review_routing import Refresh, Routing, refresh_reconciliation

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

# Rows shown in the table; the counts above it cover every extraction.
TABLE_ROWS = 40


@dataclass(frozen=True)
class Outcome:
    """What one extraction's refresh did, reduced to what the report prints."""

    filename: str
    status: str
    created: bool
    failing: tuple[str, ...]
    changes: tuple[str, ...]


def _flag(routing: Routing) -> str:
    return routing.review_reason or "-"


def _summarize(filename: str, refresh: Refresh) -> Outcome:
    return Outcome(
        filename=filename,
        status=refresh.report.status,
        created=refresh.created,
        failing=tuple(check.id for check in refresh.report.failing_checks),
        changes=tuple(
            f"{c.field_name} {_flag(c.before)} -> {_flag(c.after)}" for c in refresh.changes
        ),
    )


async def reconcile_all(
    session_factory: SessionFactory, settings: Settings, *, dry_run: bool
) -> tuple[list[Outcome], list[str]]:
    """Refresh every extraction, each in its own transaction (committed, or
    rolled back on a dry run). Returns an Outcome per extraction that worked
    and an error line per extraction that did not."""
    stmt = (
        select(Extraction.id, Document.filename)
        .join(Document, Extraction.document_id == Document.id)
        .order_by(Extraction.created_at, Extraction.id)
    )
    async with session_factory() as session:
        rows = [tuple(row) for row in (await session.execute(stmt)).all()]
    done: list[Outcome] = []
    errors: list[str] = []
    for extraction_id, filename in rows:
        async with session_factory() as session:
            try:
                outcome = _summarize(
                    filename, await refresh_reconciliation(session, extraction_id, settings)
                )
                if dry_run:
                    await session.rollback()
                else:
                    await session.commit()
                done.append(outcome)
            except Exception as exc:  # noqa: BLE001 -- one bad extraction must not stop the rest
                await session.rollback()
                errors.append(f"{filename} ({extraction_id}): {type(exc).__name__}: {exc}")
    return done, errors


def render(done: list[Outcome], errors: list[str], *, dry_run: bool) -> str:
    statuses = Counter(o.status for o in done)
    changed = [o for o in done if o.changes]
    lines = [
        (
            f"{'Dry run: nothing written. ' if dry_run else ''}"
            f"{len(done)} extractions reconciled, {sum(o.created for o in done)} new reports"
        ),
        "status: " + ", ".join(f"{status} {statuses[status]}" for status in ("pass", "fail", "skipped")),
        f"routing changes: {sum(len(o.changes) for o in done)} fields in {len(changed)} extractions",
    ]
    interesting = [o for o in done if o.changes or o.status == "fail"]
    if interesting:
        lines.append("")
        lines.append(f"{'document':<42} {'status':<7} failing checks / routing changes")
        for o in interesting[:TABLE_ROWS]:
            changes = "; ".join(o.changes) or "no routing change"
            lines.append(f"{o.filename[:41]:<42} {o.status:<7} {', '.join(o.failing) or '-'} | {changes}")
        if len(interesting) > TABLE_ROWS:
            lines.append(f"... and {len(interesting) - TABLE_ROWS} more")
    for error in errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


async def main(dry_run: bool) -> int:
    try:
        done, errors = await reconcile_all(async_session_maker, get_settings(), dry_run=dry_run)
        print(render(done, errors, dry_run=dry_run))
        if not dry_run:
            print(f"Committed {len(done)} reconciliations.")
        return 1 if errors else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="compute and print, but write nothing"
    )
    sys.exit(asyncio.run(main(parser.parse_args().dry_run)))
