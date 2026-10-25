"""Several workers draining one queue (app/worker.py), as the runtime role under
row-level security and across two organizations: each job is processed once
(no double processing, no lost job), and what each handler wrote lands once,
in its job's organization.

The workers are asyncio tasks, each looping run_once on its own sessions and
connections, which is what separate worker processes do to the database: the
only thing they share is the table. Every row is committed for real and removed
in a finally. Handlers are fakes; no model is called."""

import asyncio
import uuid
from collections import Counter
from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy import delete, func, select, text

from app.db import async_session_maker
from app.extraction import NonRetryableExtractionError
from app.models import DEFAULT_ORGANIZATION_ID, Document, Job, Organization
from app.tenancy import tenant_session
from app.worker import run_once

WORKERS = 4
JOBS_PER_ORGANIZATION = 12


@pytest_asyncio.fixture
async def committed_organization() -> AsyncGenerator[Organization]:
    organization = Organization(name="Scaling test", slug=f"scaling-{uuid.uuid4().hex[:8]}")
    async with async_session_maker() as session:
        session.add(organization)
        await session.commit()
    yield organization
    async with tenant_session(async_session_maker, organization.id) as session:
        await session.execute(delete(Job))
        await session.execute(delete(Document))
        await session.commit()
    async with async_session_maker() as session:
        await session.execute(delete(Organization).where(Organization.id == organization.id))
        await session.commit()


async def _queue_jobs(organization_id: uuid.UUID, marker: str) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """One document with JOBS_PER_ORGANIZATION pending jobs, committed."""
    async with tenant_session(async_session_maker, organization_id) as session:
        document = Document(
            filename=f"{marker}-source", mime_type="image/png", storage_path="/tmp/s", status="uploaded"
        )
        session.add(document)
        await session.flush()
        jobs = [Job(document_id=document.id, state="pending") for _ in range(JOBS_PER_ORGANIZATION)]
        session.add_all(jobs)
        await session.commit()
        return document.id, [job.id for job in jobs]


async def _drain(handler, document_ids: list[uuid.UUID]) -> int:
    """One worker: claim and run jobs until none of the documents' is pending."""
    processed = 0
    while True:
        claimed = [await run_once(handler, document_id=document_id) for document_id in document_ids]
        if not any(claimed):
            return processed
        processed += sum(claimed)


async def test_workers_racing_a_backlog_in_two_organizations_process_each_job_exactly_once(
    committed_organization: Organization,
) -> None:
    marker = f"scale-{uuid.uuid4().hex[:12]}"
    organizations = [DEFAULT_ORGANIZATION_ID, committed_organization.id]
    queued: dict[uuid.UUID, tuple[uuid.UUID, list[uuid.UUID]]] = {}
    handled: list[uuid.UUID] = []
    handled_for: dict[uuid.UUID, uuid.UUID] = {}

    async def handler(session, job) -> None:
        handled.append(job.id)
        handled_for[job.id] = job.organization_id
        session.add(
            Document(
                filename=f"{marker}-wrote-{job.id}", mime_type="image/png", storage_path="/tmp/s", status="uploaded"
            )
        )
        await session.flush()
        # Yield the loop mid-job so the workers' claims and completions interleave.
        await asyncio.sleep(0.005)

    try:
        for organization_id in organizations:
            queued[organization_id] = await _queue_jobs(organization_id, marker)
        document_ids = [document_id for document_id, _ in queued.values()]
        all_job_ids = {job_id for _, job_ids in queued.values() for job_id in job_ids}

        async with async_session_maker() as session:
            role = (
                await session.execute(
                    text("SELECT rolbypassrls OR rolsuper FROM pg_roles WHERE rolname = current_user")
                )
            ).scalar_one()
        assert role is False  # the runtime role: row-level security binds every claim and write

        per_worker = await asyncio.gather(*(_drain(handler, document_ids) for _ in range(WORKERS)))

        assert Counter(handled) == dict.fromkeys(all_job_ids, 1)  # none twice, none lost
        assert sum(per_worker) == len(all_job_ids)
        assert sum(1 for count in per_worker if count) > 1  # the backlog really was shared

        for organization_id, (document_id, job_ids) in queued.items():
            async with tenant_session(async_session_maker, organization_id) as session:
                jobs = (await session.execute(select(Job).where(Job.document_id == document_id))).scalars().all()
                assert {job.id for job in jobs} == set(job_ids)
                assert all(job.state == "done" and job.attempts == 1 for job in jobs)
                assert all(job.finished_at is not None for job in jobs)
                written = set(
                    (
                        await session.execute(
                            select(Document.filename).where(Document.filename.like(f"{marker}-wrote-%"))
                        )
                    ).scalars()
                )
                assert written == {f"{marker}-wrote-{job_id}" for job_id in job_ids}
                assert {handled_for[job_id] for job_id in job_ids} == {organization_id}
    finally:
        for organization_id in organizations:
            async with tenant_session(async_session_maker, organization_id) as session:
                if organization_id in queued:
                    document_id, _ = queued[organization_id]
                    await session.execute(delete(Job).where(Job.document_id == document_id))
                await session.execute(delete(Document).where(Document.filename.like(f"{marker}-%")))
                await session.commit()


async def test_a_worker_that_fails_a_job_mid_race_does_not_disturb_the_others(
    committed_organization: Organization,
) -> None:
    """One job's handler raises for good while the workers race: it alone is
    'failed', and every other job is done once."""
    marker = f"scale-{uuid.uuid4().hex[:12]}"
    document_id, job_ids = await _queue_jobs(committed_organization.id, marker)
    doomed = job_ids[3]
    handled: list[uuid.UUID] = []

    async def handler(session, job) -> None:
        handled.append(job.id)
        await asyncio.sleep(0.005)
        if job.id == doomed:
            raise NonRetryableExtractionError("bad file")

    try:
        await asyncio.gather(*(_drain(handler, [document_id]) for _ in range(WORKERS)))

        assert Counter(handled) == dict.fromkeys(job_ids, 1)
        async with tenant_session(async_session_maker, committed_organization.id) as session:
            states = dict(
                (
                    await session.execute(
                        select(Job.state, func.count()).where(Job.document_id == document_id).group_by(Job.state)
                    )
                ).all()
            )
            assert states == {"done": JOBS_PER_ORGANIZATION - 1, "failed": 1}
            assert (await session.get(Job, doomed)).state == "failed"
    finally:
        async with tenant_session(async_session_maker, committed_organization.id) as session:
            await session.execute(delete(Job).where(Job.document_id == document_id))
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.commit()

