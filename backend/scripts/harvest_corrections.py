"""Export fully-reviewed documents as new eval cases.

Thin CLI over app.evals.harvest -- see that module's docstring for the
rules (which documents qualify, how human truth is assembled, dedupe via
source_document_id). Run from `backend/` with the venv active and
Postgres up:

    cd backend
    python scripts/harvest_corrections.py

Re-running is safe: already-harvested documents are skipped.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import async_session_maker, engine
from app.evals.harvest import harvest_corrections


async def main() -> int:
    try:
        async with async_session_maker() as session:
            outcomes = await harvest_corrections(session)
    finally:
        await engine.dispose()

    harvested = [o for o in outcomes if o.status == "harvested"]
    for outcome in outcomes:
        detail = f" ({outcome.detail})" if outcome.detail else ""
        target = f" -> {outcome.doc_id}" if outcome.doc_id else ""
        print(f"{outcome.document_id}: {outcome.status}{target}{detail}")
    print(
        f"\n{len(harvested)} new eval case(s) from {len(outcomes)} candidate document(s)."
    )
    if harvested:
        print("Review the new files under evals/labels/ and evals/docs/, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
