"""Export fully-reviewed documents as new eval cases.

Thin CLI over app.evals.harvest -- see that module's docstring for the
rules (which documents qualify, how human truth is assembled, dedupe via
source_document_id). Run from `backend/` with the venv active and
Postgres up:

    cd backend
    python scripts/harvest_corrections.py --organization acme

It exports one organization's documents into this repository's labels, which
are public, so the organization is always named: there is no default and no
"all".

Re-running is safe: already-harvested documents are skipped.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import async_session_maker, engine
from app.evals.harvest import harvest_corrections
from app.tenancy import organization_ids, tenant_session


async def main(organization: str) -> int:
    try:
        async with async_session_maker() as session:
            try:
                (organization_id,) = await organization_ids(session, organization)
            except ValueError as exc:  # no organization with that slug
                print(f"error: {exc}", file=sys.stderr)
                return 1
        async with tenant_session(async_session_maker, organization_id) as session:
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--organization",
        required=True,
        metavar="SLUG",
        help="the organization whose reviewed documents become labels",
    )
    sys.exit(asyncio.run(main(parser.parse_args().organization)))
