"""Application-layer tenancy (app/tenancy.py): every query runs for one
organization, and a session with none refuses.

The suite's database role owns the tables and is a superuser, so nothing the
database does (row-level security) applies here: these tests show the ORM layer
holds on its own. Most run in db_session's rolled-back transaction, with the
default organization and `second_organization`; the worker test commits real
rows and removes them again.
"""

import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import (
    delete,
    event,
    func,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import aliased

from app import worker as worker_module
from app.agent.tools import TOOL_HANDLERS, ToolContext, execute_tool
from app.auth.principal import Principal, current_principal
from app.db import (
    async_session_maker,
    engine,
    get_session,
    get_tenant_session_factory,
)
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    EMBEDDING_DIM,
    AskRun,
    Base,
    Document,
    DocumentChunk,
    DocumentPage,
    ExtractedField,
    Extraction,
    Job,
    Organization,
    Reconciliation,
    TenantMixin,
)
from app.retrieval.search import search
from app.tenancy import (
    QUEUE_KEY,
    TENANT_KEY,
    TenantScopeError,
    bind_tenant,
    ephemeral_organization,
    organization_ids,
    queue_session,
    tenant_session,
)
from app.worker import claim_job, run_once

CONTENT_MODELS = [Document, Job, Extraction, ExtractedField, Reconciliation, DocumentPage, DocumentChunk, AskRun]
UNIT_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIM - 1)


async def _populate(session: AsyncSession, tag: str) -> dict[type, uuid.UUID]:
    """One row of each content model in the organization the session is bound
    to, parents first (the composite foreign keys need them). Returns each
    row's id by model."""
    document = Document(
        filename=f"{tag}.png", mime_type="image/png", storage_path=f"/tmp/{tag}", status="extracted"
    )
    session.add(document)
    await session.flush()
    ask_run = AskRun(
        question=f"{tag}?",
        status="answered",
        answer="yes",
        citations=[],
        tool_calls=[],
        evidence="",
        steps=1,
        model="m",
        prompt_version="v",
    )
    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="m",
        raw_response={},
        input_tokens=1,
        output_tokens=1,
        cost_usd=1,
        latency_ms=1,
    )
    page = DocumentPage(document_id=document.id, page_number=1, text=tag, source="gold")
    session.add_all([ask_run, extraction, page])
    await session.flush()
    session.add_all(
        [
            ExtractedField(extraction_id=extraction.id, field_name="total", value={}, confidence=0.9),
            Reconciliation(extraction_id=extraction.id, rules_version="v", status="pass", report={}),
            DocumentChunk(
                document_id=document.id,
                page_id=page.id,
                page_number=1,
                chunk_index=0,
                text=tag,
                char_start=0,
                char_end=len(tag),
                embedding=UNIT_VECTOR,
                embedding_model="m",
            ),
            Job(document_id=document.id, state="done"),
        ]
    )
    await session.flush()
    ids = {Document: document.id, AskRun: ask_run.id, Extraction: extraction.id, DocumentPage: page.id}
    for model, column in (
        (ExtractedField, "extraction_id"),
        (Reconciliation, "extraction_id"),
        (DocumentChunk, "document_id"),
        (Job, "document_id"),
    ):
        parent = {"extraction_id": extraction.id, "document_id": document.id}[column]
        ids[model] = (
            await session.execute(select(model.id).where(getattr(model, column) == parent))
        ).scalar_one()
    return ids


@pytest.fixture
async def two_tenants(
    db_session: AsyncSession, second_organization: Organization
) -> tuple[dict[type, uuid.UUID], dict[type, uuid.UUID]]:
    """A row of every content model in the default organization and in the
    second one. The session ends up bound to the default organization with an
    empty identity map, so what a test reads comes from the database."""
    await bind_tenant(db_session, second_organization.id)
    theirs = await _populate(db_session, "theirs")
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    ours = await _populate(db_session, "ours")
    await bind_tenant(db_session, second_organization.id)
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    return ours, theirs


@asynccontextmanager
async def recorded_statements() -> AsyncGenerator[list[str]]:
    """Every statement the engine sends to the database while the block runs."""
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)


# --- the schema --------------------------------------------------------------


def test_every_content_model_carries_an_organization() -> None:
    assert set(TenantMixin.__subclasses__()) == set(CONTENT_MODELS)
    for model in CONTENT_MODELS:
        column = model.__table__.c.organization_id
        assert column.nullable is False
        assert "docpilot_current_organization_id()" in str(column.server_default.arg)
    # The rest is identity: no organization column anywhere else.
    others = {t.name for t in Base.metadata.tables.values() if "organization_id" in t.c}
    assert others - {m.__tablename__ for m in CONTENT_MODELS} == {"memberships", "auth_sessions", "api_keys"}


async def test_the_models_and_the_migration_describe_the_same_database() -> None:
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    def differences(connection):
        return compare_metadata(MigrationContext.configure(connection), Base.metadata)

    async with engine.connect() as connection:
        assert await connection.run_sync(differences) == []


async def test_without_a_tenant_the_database_default_is_null_and_refuses_the_row(
    db_session: AsyncSession,
) -> None:
    assert (
        await db_session.execute(text("SELECT docpilot_current_organization_id() IS NULL"))
    ).scalar_one() is False  # db_session is bound
    async with engine.connect() as connection:
        transaction = await connection.begin()
        assert (await connection.execute(text("SELECT docpilot_current_organization_id()"))).scalar_one() is None
        with pytest.raises(IntegrityError, match="organization_id"):
            await connection.execute(
                text("INSERT INTO documents (id, filename, mime_type, storage_path, status) "
                     "VALUES (gen_random_uuid(), 'f', 'm', 'p', 'uploaded')")
            )
        await transaction.rollback()


async def test_the_function_reads_the_setting_and_treats_blank_as_unset(db_session: AsyncSession) -> None:
    async def current() -> uuid.UUID | None:
        return (await db_session.execute(text("SELECT docpilot_current_organization_id()"))).scalar_one()

    assert await current() == DEFAULT_ORGANIZATION_ID
    await db_session.execute(text("SELECT set_config('docpilot.organization_id', '', true)"))
    assert await current() is None


# --- a bound session sees one organization -----------------------------------


@pytest.mark.parametrize("model", CONTENT_MODELS)
async def test_a_select_sees_only_the_bound_organizations_rows(db_session, two_tenants, model) -> None:
    ours, theirs = two_tenants

    found = (
        await db_session.execute(select(model.id).where(model.id.in_([ours[model], theirs[model]])))
    ).scalars().all()

    assert found == [ours[model]]


@pytest.mark.parametrize("model", CONTENT_MODELS)
async def test_a_count_get_and_alias_see_only_the_bound_organizations_rows(
    db_session, second_organization, two_tenants, model
) -> None:
    ours, theirs = two_tenants
    await bind_tenant(db_session, second_organization.id)

    assert (await db_session.execute(select(func.count()).select_from(model))).scalar_one() == 1
    assert await db_session.get(model, ours[model]) is None
    assert (await db_session.get(model, theirs[model])).id == theirs[model]
    alias = aliased(model)
    assert (await db_session.execute(select(alias.id).where(alias.id == ours[model]))).all() == []
    assert (
        await db_session.execute(select(model.id).where(select(model.id).where(model.id == ours[model]).exists()))
    ).all() == []


@pytest.mark.parametrize("model", CONTENT_MODELS)
async def test_an_update_and_a_delete_touch_only_the_bound_organizations_rows(
    db_session, second_organization, two_tenants, model
) -> None:
    ours, theirs = two_tenants
    no_op = {model.__table__.c.id: model.__table__.c.id}  # SET id = id: matches rows, changes none

    assert (await db_session.execute(update(model).where(model.id == theirs[model]).values(no_op))).rowcount == 0
    assert (await db_session.execute(delete(model).where(model.id == theirs[model]))).rowcount == 0
    assert (await db_session.execute(update(model).where(model.id == ours[model]).values(no_op))).rowcount == 1

    await bind_tenant(db_session, second_organization.id)
    assert (await db_session.execute(select(model.id).where(model.id == theirs[model]))).scalar_one() == theirs[model]


async def test_a_join_and_a_subquery_are_scoped_on_every_table_they_name(
    db_session, second_organization, two_tenants
) -> None:
    ours, theirs = two_tenants
    joined = select(Extraction.id).join(Document, Document.id == Extraction.document_id)
    inside = select(Extraction.id).where(
        Extraction.document_id.in_(select(Document.id).where(Document.id.in_([ours[Document], theirs[Document]])))
    )

    assert (await db_session.execute(joined.where(Extraction.id.in_([ours[Extraction], theirs[Extraction]])))).scalars().all() == [ours[Extraction]]
    assert (await db_session.execute(inside)).scalars().all() == [ours[Extraction]]

    await bind_tenant(db_session, second_organization.id)
    assert (await db_session.execute(joined)).scalars().all() == [theirs[Extraction]]


async def test_switching_organizations_empties_the_identity_map(
    db_session, second_organization, two_tenants
) -> None:
    ours, _ = two_tenants
    mine = await db_session.get(Document, ours[Document])
    assert mine in db_session

    await bind_tenant(db_session, second_organization.id)

    # Without this, get() would hand back the other organization's object from
    # the identity map, and a refresh would reload it, as the ORM's criteria
    # do not apply to either.
    assert mine not in db_session
    assert await db_session.get(Document, ours[Document]) is None


async def test_rebinding_the_same_organization_keeps_the_session_as_it_is(db_session) -> None:
    document = Document(filename="f", mime_type="m", storage_path="p")
    db_session.add(document)

    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)

    assert document in db_session


# --- writes take the bound organization --------------------------------------


@pytest.mark.parametrize("which", ["default", "second"])
async def test_inserts_take_the_bound_organization_and_the_session_sees_it_at_once(
    db_session, second_organization, which
) -> None:
    organization_id = DEFAULT_ORGANIZATION_ID if which == "default" else second_organization.id
    await bind_tenant(db_session, organization_id)

    ids = await _populate(db_session, "fresh")

    for model, row_id in ids.items():
        row = await db_session.get(model, row_id)
        # Filled by the database default and read back by the INSERT itself:
        # no refresh, which an AsyncSession could not do lazily.
        assert "organization_id" in inspect(row).dict, model
        assert row.organization_id == organization_id, model


async def test_a_bulk_insert_takes_the_bound_organization(db_session, second_organization) -> None:
    await bind_tenant(db_session, second_organization.id)

    result = await db_session.execute(
        insert(Document).returning(Document.organization_id),
        [{"filename": "f", "mime_type": "m", "storage_path": "p"}] * 2,
    )

    assert result.scalars().all() == [second_organization.id] * 2


async def test_a_row_built_for_another_organization_is_refused_before_it_is_flushed(
    db_session, second_organization
) -> None:
    db_session.add(
        Document(
            filename="f",
            mime_type="m",
            storage_path="p",
            organization_id=second_organization.id,
        )
    )

    with pytest.raises(TenantScopeError, match="another organization"):
        await db_session.flush()


async def test_a_composite_foreign_key_refuses_a_child_of_another_organizations_parent(
    db_session, second_organization, two_tenants
) -> None:
    ours, theirs = two_tenants
    # An extraction that has no reconciliation yet: the unique key on
    # extraction_id would otherwise be hit before the foreign key is checked.
    bare = Extraction(
        document_id=ours[Document], prompt_version="p", model="m", raw_response={},
        input_tokens=1, output_tokens=1, cost_usd=1, latency_ms=1,
    )
    db_session.add(bare)
    await db_session.flush()
    bare_id = bare.id
    await bind_tenant(db_session, second_organization.id)
    page_of_ours = ours[DocumentPage]

    def child(kind: str):
        fields = {
            "extraction": lambda: Extraction(
                document_id=ours[Document], prompt_version="p", model="m", raw_response={},
                input_tokens=1, output_tokens=1, cost_usd=1, latency_ms=1,
            ),
            "extracted_field": lambda: ExtractedField(
                extraction_id=ours[Extraction], field_name="total", value={}, confidence=0.9
            ),
            "reconciliation": lambda: Reconciliation(
                extraction_id=bare_id, rules_version="v", status="pass", report={}
            ),
            "page": lambda: DocumentPage(document_id=ours[Document], page_number=2, text="t", source="gold"),
            "chunk_document": lambda: DocumentChunk(
                document_id=ours[Document], page_id=theirs[DocumentPage], page_number=1, chunk_index=1,
                text="t", char_start=0, char_end=1, embedding=UNIT_VECTOR, embedding_model="m",
            ),
            "chunk_page": lambda: DocumentChunk(
                document_id=theirs[Document], page_id=page_of_ours, page_number=1, chunk_index=2,
                text="t", char_start=0, char_end=1, embedding=UNIT_VECTOR, embedding_model="m",
            ),
            "ask_run": lambda: AskRun(
                question="q", status="answered", answer="a", citations=[], tool_calls=[], evidence="",
                steps=1, model="m", prompt_version="v", document_id=ours[Document], conversation_id=uuid.uuid4(),
            ),
            "job_document": lambda: Job(document_id=ours[Document], state="pending"),
            "job_ask_run": lambda: Job(ask_run_id=ours[AskRun], state="pending"),
        }
        return fields[kind]()

    for kind, constraint in (
        ("extraction", "extractions_document_id_organization_id_fkey"),
        ("extracted_field", "extracted_fields_extraction_id_organization_id_fkey"),
        ("reconciliation", "reconciliations_extraction_id_organization_id_fkey"),
        ("page", "document_pages_document_id_organization_id_fkey"),
        ("chunk_document", "document_chunks_document_id_organization_id_fkey"),
        ("chunk_page", "document_chunks_page_id_organization_id_fkey"),
        ("ask_run", "ask_runs_document_id_organization_id_fkey"),
        ("job_document", "jobs_document_id_organization_id_fkey"),
        ("job_ask_run", "jobs_ask_run_id_organization_id_fkey"),
    ):
        with pytest.raises(IntegrityError, match=constraint):
            async with db_session.begin_nested():
                db_session.add(child(kind))
                await db_session.flush()


async def test_the_same_children_are_accepted_in_their_own_organization(db_session, two_tenants) -> None:
    ours, _ = two_tenants

    async with db_session.begin_nested():
        db_session.add(Job(document_id=ours[Document], state="pending"))
        db_session.add(Job(ask_run_id=ours[AskRun], state="pending"))
        await db_session.flush()


async def test_a_job_with_neither_a_document_nor_an_ask_run_is_still_refused(db_session) -> None:
    with pytest.raises(IntegrityError, match="ck_jobs_has_target"):
        async with db_session.begin_nested():
            db_session.add(Job(state="pending"))
            await db_session.flush()


async def test_the_original_on_delete_behavior_is_kept(db_session, two_tenants) -> None:
    ours, _ = two_tenants
    job = Job(ask_run_id=ours[AskRun], state="pending")
    db_session.add(job)
    await db_session.flush()
    job_id = job.id

    # jobs.ask_run_id cascaded before and still does.
    await db_session.execute(delete(AskRun).where(AskRun.id == ours[AskRun]))
    assert (await db_session.execute(select(Job.id).where(Job.id == job_id))).first() is None

    # reconciliations.extraction_id cascaded; extracted_fields.extraction_id and
    # every document_id never did.
    with pytest.raises(IntegrityError, match="extracted_fields_extraction_id_organization_id_fkey"):
        async with db_session.begin_nested():
            await db_session.execute(delete(Extraction).where(Extraction.id == ours[Extraction]))
    await db_session.execute(delete(ExtractedField).where(ExtractedField.id == ours[ExtractedField]))
    await db_session.execute(delete(Extraction).where(Extraction.id == ours[Extraction]))
    assert (
        await db_session.execute(select(Reconciliation.id).where(Reconciliation.id == ours[Reconciliation]))
    ).first() is None
    with pytest.raises(IntegrityError, match="violates foreign key constraint"):
        async with db_session.begin_nested():
            await db_session.execute(delete(Document).where(Document.id == ours[Document]))


@pytest.mark.parametrize("model", CONTENT_MODELS)
async def test_a_union_sees_only_the_bound_organizations_rows(
    db_session, second_organization, two_tenants, model
) -> None:
    ours, theirs = two_tenants
    ids = [ours[model], theirs[model]]
    branch = select(model.id).where(model.id.in_(ids))

    assert (await db_session.execute(branch.union(branch))).scalars().all() == [ours[model]]
    assert (await db_session.execute(branch.union_all(branch))).scalars().all() == [ours[model]] * 2
    assert (await db_session.execute(branch.union(select(model.id).where(model.id.in_(ids))))).all() == [(ours[model],)]

    await bind_tenant(db_session, second_organization.id)
    assert (await db_session.execute(branch.union_all(branch))).scalars().all() == [theirs[model]] * 2


@pytest.mark.parametrize(
    "build",
    [
        lambda theirs, other: select(Document).from_statement(text("SELECT * FROM documents")),
        lambda theirs, other: select(Document.__table__),
        lambda theirs, other: select(DocumentChunk.__table__.c.id),
        lambda theirs, other: delete(DocumentChunk.__table__).where(DocumentChunk.__table__.c.text == "x"),
        lambda theirs, other: update(Document.__table__).values(status="x"),
        lambda theirs, other: insert(Document.__table__).values(filename="f", mime_type="m", storage_path="p"),
        lambda theirs, other: insert(Document).from_select(
            ["filename", "mime_type", "storage_path"], select(Document.filename, Document.mime_type, Document.storage_path)
        ),
        lambda theirs, other: insert(Document).values(
            filename="f", mime_type="m", storage_path="p", organization_id=other
        ),
        lambda theirs, other: insert(Document).values(
            [{"filename": "f", "mime_type": "m", "storage_path": "p", "organization_id": other}]
        ),
        lambda theirs, other: update(Document).where(Document.id == theirs).values(organization_id=other),
        lambda theirs, other: update(Document).values({Document.organization_id: other}),
    ],
    ids=[
        "from-statement", "core-select", "core-columns", "core-delete", "core-update", "core-insert",
        "insert-from-select", "insert-org", "insert-many-org", "update-org",
        "update-org-attribute",
    ],
)
async def test_a_form_the_filter_cannot_cover_is_refused_on_a_bound_session(
    db_session, second_organization, two_tenants, build
) -> None:
    _, theirs = two_tenants
    statement = build(theirs[Document], second_organization.id)

    async with recorded_statements() as sent:
        with pytest.raises(TenantScopeError):
            await db_session.execute(statement)
    assert sent == []


@pytest.mark.parametrize(
    ("statement", "rows"),
    [
        (lambda: update(Document), lambda theirs, other: [{"id": theirs, "status": "x"}]),
        (lambda: delete(Document), lambda theirs, other: [{"id": theirs}]),
        (lambda: insert(Document), lambda theirs, other: [
            {"filename": "f", "mime_type": "m", "storage_path": "p", "organization_id": other}
        ]),
    ],
    ids=["bulk-update", "bulk-delete", "bulk-insert-org"],
)
async def test_a_bulk_statement_by_primary_key_or_with_an_organization_is_refused(
    db_session, second_organization, two_tenants, statement, rows
) -> None:
    _, theirs = two_tenants

    with pytest.raises(TenantScopeError):
        await db_session.execute(statement(), rows(theirs[Document], second_organization.id))

    await bind_tenant(db_session, second_organization.id)
    stored = (await db_session.execute(select(Document.status).where(Document.id == theirs[Document]))).scalar_one()
    assert stored == "extracted"  # the other organization's row was not touched


async def test_an_ordinary_bulk_insert_and_a_filtered_bulk_update_are_still_fine(db_session) -> None:
    await db_session.execute(
        insert(Document), [{"filename": "f", "mime_type": "m", "storage_path": "p"}] * 2
    )
    updated = await db_session.execute(update(Document).where(Document.filename == "f").values(status="x"))
    assert updated.rowcount >= 2


async def test_changing_a_loaded_rows_organization_is_refused(db_session, second_organization, two_tenants) -> None:
    ours, _ = two_tenants
    document = await db_session.get(Document, ours[Document])

    document.organization_id = second_organization.id

    with pytest.raises(TenantScopeError, match="cannot be changed"):
        await db_session.flush()


async def test_an_unchanged_organization_assignment_does_not_trip_the_guard(db_session, two_tenants) -> None:
    ours, _ = two_tenants
    document = await db_session.get(Document, ours[Document])

    document.organization_id = DEFAULT_ORGANIZATION_ID
    document.status = "reviewed"
    await db_session.flush()

    assert (await db_session.get(Document, ours[Document])).status == "reviewed"


# --- a session with no organization refuses ---------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        lambda: select(Document),
        lambda: select(Document.id).where(Document.filename == "x"),
        lambda: select(func.count()).select_from(Document),
        lambda: select(aliased(Document)),
        lambda: select(Extraction.id).join(Document, Document.id == Extraction.document_id),
        lambda: select(Organization.id).where(select(Document.id).exists()),
        lambda: select(Document.id).where(Document.id.in_(select(Extraction.document_id))),
        lambda: update(Document).values(status="x"),
        lambda: delete(Document),
        lambda: insert(Document).values(filename="f", mime_type="m", storage_path="p"),
        lambda: select(Job).with_for_update(skip_locked=True).limit(1),
        lambda: select(Document.id).union(select(Document.id)),
        lambda: select(Organization.id).union_all(select(Document.id)),
        lambda: select(Document).from_statement(text("SELECT * FROM documents")),
        lambda: select(Document.__table__),
        lambda: delete(Document.__table__),
    ],
    ids=[
        "select", "columns", "count", "alias", "join", "exists", "subquery", "update", "delete", "insert",
        "job-claim", "union", "union-all", "from-statement", "core-select", "core-delete",
    ],
)
async def test_a_statement_on_content_tables_is_refused_before_any_sql(statement) -> None:
    async with async_session_maker() as session:
        async with recorded_statements() as sent:
            with pytest.raises(TenantScopeError, match="organization"):
                await session.execute(statement())
        assert sent == []


async def test_get_and_flush_on_an_unbound_session_are_refused_before_any_sql() -> None:
    async with async_session_maker() as session:
        async with recorded_statements() as sent:
            with pytest.raises(TenantScopeError):
                await session.get(Document, uuid.uuid4())
            session.add(Document(filename="f", mime_type="m", storage_path="p"))
            with pytest.raises(TenantScopeError, match="flushed"):
                await session.flush()
            session.expunge_all()
            session.add(Job(state="pending", document_id=uuid.uuid4()))
            with pytest.raises(TenantScopeError):
                await session.commit()
        assert sent == []


async def test_statements_that_touch_no_content_table_need_no_organization(db_session) -> None:
    async with async_session_maker() as session:
        assert (await session.execute(select(func.count()).select_from(Organization))).scalar_one() >= 1
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
        assert await organization_ids(session, "default") == [DEFAULT_ORGANIZATION_ID]
        assert DEFAULT_ORGANIZATION_ID in await organization_ids(session)


async def test_a_queue_session_may_claim_jobs_of_any_organization_and_nothing_else(
    db_session, second_organization, two_tenants
) -> None:
    ours, theirs = two_tenants
    async with queue_session(lambda: _same(db_session)) as session:
        assert session.info[QUEUE_KEY] is True
        # Unbound: the queue is the one table it may read, across organizations.
        await _unbind(session)
        both = (
            await session.execute(select(Job.organization_id).where(Job.id.in_([ours[Job], theirs[Job]])))
        ).scalars().all()
        assert set(both) == {DEFAULT_ORGANIZATION_ID, second_organization.id}
        job = await session.get(Job, ours[Job])
        job.state = "processing"
        await session.flush()
        for statement in (select(Document), select(Extraction), delete(Document), update(AskRun).values(steps=2)):
            with pytest.raises(TenantScopeError, match="queue session"):
                await session.execute(statement)
        session.add(Document(filename="f", mime_type="m", storage_path="p"))
        with pytest.raises(TenantScopeError, match="flushed"):
            await session.flush()


@asynccontextmanager
async def _same(session: AsyncSession) -> AsyncGenerator[AsyncSession]:
    yield session


async def _unbind(session: AsyncSession) -> None:
    await session.flush()
    session.expunge_all()
    session.info.pop(TENANT_KEY, None)


# --- the organization follows every transaction ------------------------------


async def _setting(session: AsyncSession) -> str | None:
    return (await session.execute(text("SELECT current_setting('docpilot.organization_id', true)"))).scalar_one()


async def test_the_organization_is_reapplied_after_a_commit_and_after_a_rollback(
    db_session, second_organization
) -> None:
    for organization_id in (DEFAULT_ORGANIZATION_ID, second_organization.id):
        await bind_tenant(db_session, organization_id)
        assert await _setting(db_session) == str(organization_id)
        await db_session.commit()
        assert await _setting(db_session) == str(organization_id)
        await db_session.rollback()
        assert await _setting(db_session) == str(organization_id)
        db_session.add(Document(filename="f", mime_type="m", storage_path="p"))
        await db_session.flush()
        document = (await db_session.execute(select(Document).order_by(Document.created_at.desc()).limit(1))).scalar_one()
        assert document.organization_id == organization_id
        await db_session.rollback()


async def test_the_setting_does_not_leak_to_the_next_user_of_a_pooled_connection() -> None:
    # One connection in the pool and no overflow: every session below, the
    # tenant's and the next user's, runs on that same connection, so a setting
    # that outlived its transaction would be seen by the next user.
    dedicated = create_async_engine(engine.url, pool_size=1, max_overflow=0)
    factory = async_sessionmaker(dedicated, expire_on_commit=False)
    organization_id = uuid.uuid4()  # never stored: only the setting is read

    async def next_user_sees() -> str | None:
        async with factory() as session:
            return await _setting(session)

    try:
        async with tenant_session(factory, organization_id) as session:
            assert await _setting(session) == str(organization_id)
            await session.commit()
            assert await next_user_sees() == ""  # after a commit

            assert await _setting(session) == str(organization_id)  # a new transaction has it again
            await session.rollback()
            assert await next_user_sees() == ""  # after a rollback

            assert await _setting(session) == str(organization_id)
        assert await next_user_sees() == ""  # after the session closed without committing
        assert dedicated.pool.checkedout() == 0
    finally:
        await dedicated.dispose()


async def test_binding_while_a_transaction_is_open_applies_at_once(db_session, second_organization) -> None:
    assert await _setting(db_session) == str(DEFAULT_ORGANIZATION_ID)

    await bind_tenant(db_session, second_organization.id)

    assert await _setting(db_session) == str(second_organization.id)


async def test_the_organization_survives_a_tool_savepoint_that_failed(
    db_session, second_organization, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.retrieval.embeddings import HashingEmbedder

    async def broken(ctx, tool_input):
        await ctx.session.execute(text("SELECT 1 / 0"))

    monkeypatch.setitem(TOOL_HANDLERS, "query_extractions", broken)
    await bind_tenant(db_session, second_organization.id)
    context = ToolContext(session=db_session, embedder=HashingEmbedder())

    outcome = await execute_tool(context, "query_extractions", {}, "call-1")

    assert outcome.is_error is True
    assert await _setting(db_session) == str(second_organization.id)
    db_session.add(Document(filename="f", mime_type="m", storage_path="p"))
    await db_session.flush()
    stored = (await db_session.execute(select(Document).where(Document.filename == "f"))).scalar_one()
    assert stored.organization_id == second_organization.id


async def test_the_organization_survives_a_tool_savepoint_that_failed_on_a_real_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.retrieval.embeddings import HashingEmbedder

    async def broken(ctx, tool_input):
        await ctx.session.execute(text("SELECT 1 / 0"))

    monkeypatch.setitem(TOOL_HANDLERS, "query_extractions", broken)
    async with tenant_session(async_session_maker, DEFAULT_ORGANIZATION_ID) as session:
        outcome = await execute_tool(
            ToolContext(session=session, embedder=HashingEmbedder()), "query_extractions", {}, "call-1"
        )
        assert outcome.is_error is True
        assert await _setting(session) == str(DEFAULT_ORGANIZATION_ID)
        session.add(Document(filename="f", mime_type="m", storage_path="p"))
        await session.flush()  # the default fills it from the surviving setting
        await session.rollback()


# --- the entry points --------------------------------------------------------


async def test_the_placeholder_principal_is_the_default_organization() -> None:
    assert await current_principal() == Principal(DEFAULT_ORGANIZATION_ID)


async def test_get_session_hands_out_a_session_bound_to_the_principals_organization() -> None:
    generator = get_session(Principal(DEFAULT_ORGANIZATION_ID))
    session = await anext(generator)
    try:
        assert session.info[TENANT_KEY] == DEFAULT_ORGANIZATION_ID
        assert (await session.execute(select(func.count()).select_from(Document))).scalar_one() >= 0
    finally:
        await generator.aclose()


async def test_the_stream_session_factory_yields_sessions_bound_to_the_principals_organization() -> None:
    factory = get_tenant_session_factory(Principal(DEFAULT_ORGANIZATION_ID))

    async with factory() as first, factory() as second:
        assert first is not second
        assert first.info[TENANT_KEY] == second.info[TENANT_KEY] == DEFAULT_ORGANIZATION_ID


async def test_an_ephemeral_organization_exists_only_in_the_callers_transaction(db_session) -> None:
    organization = await ephemeral_organization(db_session)

    assert organization.slug.startswith("ephemeral-")
    assert db_session.info[TENANT_KEY] == organization.id
    assert await _setting(db_session) == str(organization.id)
    db_session.add(Document(filename="f", mime_type="m", storage_path="p"))
    await db_session.flush()
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 1


async def test_organization_ids_lists_every_organization_or_the_one_named(
    db_session, second_organization
) -> None:
    everyone = await organization_ids(db_session)

    assert DEFAULT_ORGANIZATION_ID in everyone and second_organization.id in everyone
    assert await organization_ids(db_session, second_organization.slug) == [second_organization.id]
    with pytest.raises(ValueError, match="no organization with slug 'nowhere'"):
        await organization_ids(db_session, "nowhere")


# --- search ------------------------------------------------------------------


class _QueryEmbedder:
    name = "fixed"
    dim = EMBEDDING_DIM

    def embed_query(self, text: str) -> list[float]:
        return UNIT_VECTOR


def _near(noise: float, index: int) -> list[float]:
    """A unit vector at a fixed angle from UNIT_VECTOR: a different, tiny
    offset per chunk keeps them from being identical."""
    vector = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    vector[1 + index % (EMBEDDING_DIM - 1)] = noise
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


async def _chunks(session: AsyncSession, count: int, vector: Callable[[int], list[float]]) -> None:
    document = Document(filename="d.png", mime_type="image/png", storage_path="/tmp/d", status="extracted")
    session.add(document)
    await session.flush()
    page = DocumentPage(document_id=document.id, page_number=1, text="t", source="gold")
    session.add(page)
    await session.flush()
    session.add_all(
        DocumentChunk(
            document_id=document.id, page_id=page.id, page_number=1, chunk_index=i, text="t",
            char_start=0, char_end=1, embedding=vector(i), embedding_model="fixed",
        )
        for i in range(count)
    )
    await session.flush()


async def test_a_small_tenant_still_fills_k_beside_a_large_one_through_the_hnsw_index(
    db_session, second_organization
) -> None:
    # Every test that indexes chunks rolls its inserts back and leaves dead
    # entries in the HNSW graph until autovacuum runs. Enough of them hide the
    # small tenant's nodes from the scan, which is a property of a bloated
    # index and not of what is tested here, so start from a rebuilt one (the
    # rebuild is part of this transaction and rolls back with it).
    await db_session.execute(text("REINDEX INDEX ix_document_chunks_embedding_hnsw"))
    # 400 chunks of the default organization sit nearer the query than the
    # second organization's 3. The HNSW scan yields ef_search candidates and
    # the tenant filter runs on them, so without an iterative scan the small
    # tenant would get nothing.
    await _chunks(db_session, 400, lambda i: _near(0.05, i))
    await bind_tenant(db_session, second_organization.id)
    await _chunks(db_session, 3, lambda i: _near(1.3, i))
    for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
        await db_session.execute(text(f"SET LOCAL {setting} = off"))

    hits = await search(db_session, "anything", embedder=_QueryEmbedder(), k=3, mode="dense")

    assert len(hits) == 3
    assert {hit.document_id for hit in hits} == {
        (await db_session.execute(select(Document.id))).scalar_one()
    }
    plan = "\n".join(
        row[0]
        for row in await db_session.execute(
            text("EXPLAIN SELECT id FROM document_chunks WHERE organization_id = :o ORDER BY embedding <=> :v LIMIT 60"),
            {"o": second_organization.id, "v": str(UNIT_VECTOR)},
        )
    )
    assert "ix_document_chunks_embedding_hnsw" in plan  # the index under test was the one used


async def test_lexical_search_counts_only_the_bound_organizations_chunks(
    db_session, second_organization
) -> None:
    from app.retrieval.embeddings import HashingEmbedder
    from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages

    embedder = HashingEmbedder()
    config = ChunkingConfig(0, 0, False)

    async def index(filename: str, text_: str) -> uuid.UUID:
        document = Document(filename=filename, mime_type="image/png", storage_path="/tmp/x", status="extracted")
        db_session.add(document)
        await db_session.flush()
        await index_document_pages(db_session, document.id, [PageText(1, text_, "gold")], embedder, config)
        return document.id

    mine = await index("mine.png", "Lamp receipt total 12.50")
    await bind_tenant(db_session, second_organization.id)
    theirs = await index("theirs.png", "Lamp receipt total 99.99")
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)

    hits = await search(db_session, "lamp receipt total", embedder=embedder, k=10, mode="lexical")

    assert {hit.document_id for hit in hits} >= {mine}
    assert theirs not in {hit.document_id for hit in hits}


# --- the worker --------------------------------------------------------------


@pytest.fixture
async def committed_second_organization() -> AsyncGenerator[Organization]:
    """A second organization committed for real (the worker opens its own
    sessions), removed with everything the test made in it."""
    organization = Organization(name="Worker test", slug=f"worker-{uuid.uuid4().hex[:8]}")
    async with async_session_maker() as session:
        session.add(organization)
        await session.commit()
    yield organization
    async with tenant_session(async_session_maker, organization.id) as session:
        for model in (ExtractedField, Reconciliation, Extraction, Job, DocumentChunk, DocumentPage, AskRun, Document):
            await session.execute(delete(model))
        await session.commit()
    async with async_session_maker() as session:
        await session.execute(delete(Organization).where(Organization.id == organization.id))
        await session.commit()


async def _committed_job(organization_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    async with tenant_session(async_session_maker, organization_id) as session:
        document = Document(filename="w.png", mime_type="image/png", storage_path="/tmp/w", status="uploaded")
        session.add(document)
        await session.flush()
        job = Job(document_id=document.id, state="pending", kind="extract")
        session.add(job)
        await session.commit()
        return document.id, job.id


async def test_the_worker_runs_each_organizations_job_under_that_organization(
    committed_second_organization: Organization,
) -> None:
    other = committed_second_organization.id
    default_document, default_job = await _committed_job(DEFAULT_ORGANIZATION_ID)
    other_document, other_job = await _committed_job(other)
    seen: list[tuple[uuid.UUID, str | None, uuid.UUID]] = []

    async def handler(session: AsyncSession, job: Job) -> None:
        extraction = Extraction(
            document_id=job.document_id, prompt_version="p", model="m", raw_response={},
            input_tokens=1, output_tokens=1, cost_usd=0, latency_ms=1,
        )
        session.add(extraction)
        await session.flush()
        seen.append((job.id, await _setting(session), extraction.organization_id))
        # The handler can reach its own document and not the other organization's.
        assert (await session.get(Document, job.document_id)) is not None
        foreign = other_document if job.document_id == default_document else default_document
        assert (await session.execute(select(Document.id).where(Document.id == foreign))).first() is None

    try:
        assert await run_once(handler, document_id=default_document) is True
        assert await run_once(handler, document_id=other_document) is True

        assert seen == [
            (default_job, str(DEFAULT_ORGANIZATION_ID), DEFAULT_ORGANIZATION_ID),
            (other_job, str(other), other),
        ]
        for organization_id, job_id, document_id in (
            (DEFAULT_ORGANIZATION_ID, default_job, default_document),
            (other, other_job, other_document),
        ):
            async with tenant_session(async_session_maker, organization_id) as session:
                job = await session.get(Job, job_id)
                assert job.state == "done" and job.organization_id == organization_id
                document = await session.get(Document, document_id)
                assert document.status == "processing"  # claimed under its own organization
    finally:
        async with tenant_session(async_session_maker, DEFAULT_ORGANIZATION_ID) as session:
            for model in (Extraction, Job):
                await session.execute(delete(model).where(model.document_id == default_document))
            await session.execute(delete(Document).where(Document.id == default_document))
            await session.commit()


async def test_a_claim_binds_the_jobs_organization_before_it_touches_the_document(
    committed_second_organization: Organization,
) -> None:
    other = committed_second_organization.id
    document_id, job_id = await _committed_job(other)

    async with queue_session(async_session_maker) as session:
        job = await claim_job(session, document_id=document_id)

        assert job is not None and job.id == job_id
        assert session.info[TENANT_KEY] == other
        assert await _setting(session) == str(other)
    async with tenant_session(async_session_maker, other) as session:
        assert (await session.get(Document, document_id)).status == "processing"
        assert (await session.get(Job, job_id)).state == "processing"


async def test_a_claim_with_nothing_pending_binds_nothing() -> None:
    async with queue_session(async_session_maker) as session:
        assert await claim_job(session, document_id=uuid.uuid4()) is None
        assert TENANT_KEY not in session.info


async def test_reclaiming_resets_jobs_of_every_organization(committed_second_organization: Organization) -> None:
    _, mine = await _committed_job(DEFAULT_ORGANIZATION_ID)
    _, theirs = await _committed_job(committed_second_organization.id)
    for organization_id, job_id in ((DEFAULT_ORGANIZATION_ID, mine), (committed_second_organization.id, theirs)):
        async with tenant_session(async_session_maker, organization_id) as session:
            (await session.get(Job, job_id)).state = "processing"
            await session.commit()

    try:
        async with queue_session(async_session_maker) as session:
            assert await worker_module.reclaim_orphaned_jobs(session) >= 2

        for organization_id, job_id in ((DEFAULT_ORGANIZATION_ID, mine), (committed_second_organization.id, theirs)):
            async with tenant_session(async_session_maker, organization_id) as session:
                assert (await session.get(Job, job_id)).state == "pending"
    finally:
        async with tenant_session(async_session_maker, DEFAULT_ORGANIZATION_ID) as session:
            job = await session.get(Job, mine)
            document_id = job.document_id
            await session.execute(delete(Job).where(Job.id == mine))
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.commit()


def test_the_queue_is_the_one_queue_scoped_model() -> None:
    assert [model for model in CONTENT_MODELS if getattr(model, "__queue_scoped__", False)] == [Job]


async def test_the_default_organization_row_is_the_one_every_old_row_was_filed_under(db_session) -> None:
    assert (await db_session.get(Organization, DEFAULT_ORGANIZATION_ID)).slug == "default"


async def test_the_daily_spend_is_the_bound_organizations_alone(
    db_session, second_organization, two_tenants
) -> None:
    from app.budget import spent_today_usd

    default_spend = await spent_today_usd(db_session)
    await bind_tenant(db_session, second_organization.id)

    # The second organization is new in this transaction: its spend is its one
    # extraction's dollar, whatever the default organization has spent today.
    assert await spent_today_usd(db_session) == pytest.approx(1.0)
    assert default_spend >= 1.0


async def test_each_savepoint_sends_the_setting_again() -> None:
    async with tenant_session(async_session_maker, DEFAULT_ORGANIZATION_ID) as session:
        await session.execute(text("SELECT 1"))  # the session's own transaction has begun
        async with recorded_statements() as sent, session.begin_nested():
            await session.execute(text("SELECT 1"))

    assert sum("set_config" in statement for statement in sent) == 1
