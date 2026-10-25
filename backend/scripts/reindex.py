"""Queue retrieval index jobs for already-extracted documents.

New uploads are indexed automatically (process_document_job enqueues an
'index' job after a successful extraction). This backfills everything
else: documents extracted before retrieval existed, or -- with --all --
every extracted document, e.g. after changing the embedding model or the
chunking settings. Each queued job re-transcribes, so --all spends one
transcription call per page (claude-haiku-4-5 by default).

It runs once per organization, in a session bound to it (app/tenancy.py):
every organization by default, or only the one named with --organization.

Run from backend/ with the venv active, then let the worker drain the
queue:

    python scripts/reindex.py            # only documents with no chunks yet
    python scripts/reindex.py --all      # every extracted document
    python scripts/reindex.py --dry-run  # just count
    python scripts/reindex.py --organization acme   # one organization, by slug
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import exists, select

from app.db import async_session_maker, engine
from app.models import JOB_KIND_INDEX, Document, DocumentChunk, Job
from app.tenancy import organization_ids, tenant_session


async def queue_index_jobs(
    *, include_indexed: bool, dry_run: bool, organization: str | None = None
) -> int:
    """Queue the jobs for each organization (or the one named) and return how
    many in all. Raises ValueError when `organization` names none."""
    async with async_session_maker() as session:
        organizations = await organization_ids(session, organization)
    count = 0
    for organization_id in organizations:
        count += await _queue_for_organization(
            organization_id, include_indexed=include_indexed, dry_run=dry_run
        )
    await engine.dispose()
    return count


async def _queue_for_organization(
    organization_id: uuid.UUID, *, include_indexed: bool, dry_run: bool
) -> int:
    async with tenant_session(async_session_maker, organization_id) as session:
        stmt = select(Document.id).where(Document.status == "extracted")
        if not include_indexed:
            stmt = stmt.where(~exists().where(DocumentChunk.document_id == Document.id))
        # Never double-queue a document that already has an index job
        # waiting or running.
        stmt = stmt.where(
            ~exists().where(
                Job.document_id == Document.id,
                Job.kind == JOB_KIND_INDEX,
                Job.state.in_(("pending", "processing")),
            )
        )
        document_ids = (await session.execute(stmt)).scalars().all()
        if not dry_run:
            session.add_all(
                Job(document_id=document_id, kind=JOB_KIND_INDEX) for document_id in document_ids
            )
            await session.commit()
    return len(document_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="re-index documents that already have chunks too"
    )
    parser.add_argument("--dry-run", action="store_true", help="count, don't queue")
    parser.add_argument(
        "--organization", metavar="SLUG", help="only this organization (default: every one)"
    )
    args = parser.parse_args()
    try:
        count = asyncio.run(
            queue_index_jobs(
                include_indexed=args.all, dry_run=args.dry_run, organization=args.organization
            )
        )
    except ValueError as exc:  # no organization with that slug
        print(f"error: {exc}", file=sys.stderr)
        return 1
    verb = "would queue" if args.dry_run else "queued"
    print(f"{verb} {count} index job(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
