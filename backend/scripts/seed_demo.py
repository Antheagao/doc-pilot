"""Load the deterministic demo dataset (app/demo_seed.py) into the database.

Replays the committed Sonnet extraction run for the 25 eval documents; no
model call is made. Normally run in the demo stack's one-shot seed service:

    backend\\.venv\\Scripts\\python scripts\\demos\\demo.py seed [--reset]

It only ever touches the demo database (docpilot_demo). An empty database
is seeded as is; a non-empty one is refused unless --reset, which replaces
every app row and upload.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import async_session_maker, engine
from app.demo_seed import (
    ensure_demo_database,
    remove_stale_uploads,
    reset_demo_data,
    seed_demo_data,
)
from app.models import Document


async def run(session: AsyncSession, settings: Settings, reset: bool) -> int:
    # Before anything is read or written: a seed pointed at the dev
    # database must stop here.
    await ensure_demo_database(session)
    if reset:
        await reset_demo_data(session, settings)
    elif (await session.execute(select(func.count()).select_from(Document))).scalar_one():
        print(
            "error: the database already has documents; run with --reset to replace them",
            file=sys.stderr,
        )
        return 1
    summary = await seed_demo_data(session, settings)
    # One transaction for the old rows going and the new ones arriving. The
    # old upload files go only after it commits (the seed rewrites the files
    # it needs in place, under the same names), so a failed seed leaves the
    # previous database and its images together.
    await session.commit()
    remove_stale_uploads(settings, summary)
    print(
        f"Seeded {len(summary.document_ids)} documents, "
        f"{summary.pending_review} fields pending review "
        f"({summary.staged} staged), "
        f"replayed cost ${summary.total_cost_usd:.4f}, "
        f"{summary.label_fallbacks} label fallbacks"
    )
    return 0


async def main(reset: bool) -> int:
    try:
        async with async_session_maker() as session:
            return await run(session, get_settings(), reset)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reset", action="store_true", help="replace every app row and upload first"
    )
    sys.exit(asyncio.run(main(parser.parse_args().reset)))
