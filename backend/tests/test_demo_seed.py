"""Tests for the deterministic demo seed (app/demo_seed.py).

The seed runs inside db_session's rolled-back transaction with the hashing
embedder and a tmp upload dir. The test database can hold rows committed by
other tests, so every assertion is scoped to the seeded document ids (or is
a before/after delta), never a global count.

The guarded entry points refuse any database but the demo one by name, and
this is docpilot_test: tests that need to get past that check patch
demo_seed.DEMO_DATABASE_NAME, and the refusal tests do not.
"""

import importlib.util
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import demo_seed
from app.auth import accounts, api_keys, passwords, sessions
from app.budget import day_start, spent_today_usd
from app.config import BACKEND_DIR, Settings
from app.demo_seed import (
    DEMO_USER_EMAIL,
    DEMO_USER_PASSWORD,
    HISTORY_DAYS,
    LABEL_REPLAY_MODEL,
    REPLAY_ARTIFACT,
    RESULTS_DIR,
    STAGED_REVIEW_CASES,
    SeedSummary,
    _wipe,
    choose_replays,
    demo_document_id,
    remove_stale_uploads,
    reset_demo_data,
    seed_demo_data,
    upsert_demo_user,
)
from app.evals.checks import label_tool_input
from app.evals.dataset import load_cases
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2, SCHEMA_V3
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    IDENTITY_TABLES,
    MEMBERSHIP_ROLE_MEMBER,
    MEMBERSHIP_ROLE_OWNER,
    ApiKey,
    AuthSession,
    Document,
    DocumentPage,
    ExtractedField,
    Extraction,
    Membership,
    Organization,
    Reconciliation,
    User,
)
from app.retrieval.embeddings import HashingEmbedder
from app.routers.documents import ALLOWED_MIME_TYPES
from app.tenancy import bind_tenant

NOW = datetime(2026, 10, 10, 15, 30, tzinfo=UTC)
STAGED = STAGED_REVIEW_CASES[0]
QUOTE_STAGED = [case for case in STAGED_REVIEW_CASES if case.doc_id.startswith("028")]
# The v2 documents the committed v1 run did not score (026-032). With the
# committed Sonnet extract_v3 run they replay that run; with no such run in
# RESULTS_DIR they replay their labels (see label_results_dir).
LABEL_DOC_IDS = sorted(case.doc_id for case in load_cases() if case.doc_id[:3] >= "026")
# The committed runs that scored them, from the paid extract_v3 smoke: the seed
# replays the Sonnet one (the primary artifact's model) and ignores the Haiku one.
V3_ARTIFACT = RESULTS_DIR / "20261017T063559Z_claude-sonnet-5_extract_v3.json"
HAIKU_V3_ARTIFACT = RESULTS_DIR / "20261017T063918Z_claude-haiku-4-5_extract_v3.json"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(upload_dir=str(tmp_path), embedding_backend="hashing")


@pytest.fixture
def artifact() -> dict:
    return json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture
def v3_artifact() -> dict:
    return json.loads(V3_ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture
def v3_entries(v3_artifact: dict) -> dict[str, dict]:
    """The scored entries of the committed Sonnet v3 run for the documents the
    seed replays from it."""
    entries = {e["doc_id"]: e for e in v3_artifact["per_doc"] if e["doc_id"] in LABEL_DOC_IDS}
    assert sorted(entries) == LABEL_DOC_IDS
    assert all(e["status"] == "scored" for e in entries.values())
    return entries


@pytest_asyncio.fixture
async def seeded(db_session: AsyncSession, settings: Settings) -> SeedSummary:
    return await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)


@pytest_asyncio.fixture
async def label_seeded(
    db_session: AsyncSession, settings: Settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> SeedSummary:
    """The seed with no committed run past the v1 one, as before extract_v3 was
    run: 026-032 replay their labels. REPLAY_ARTIFACT (v1) is a fixed path, so
    only the later runs are swapped for an empty directory."""
    empty = tmp_path.parent / f"{tmp_path.name}-no-later-runs"
    empty.mkdir()
    monkeypatch.setattr(demo_seed, "RESULTS_DIR", empty)
    return await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)


@pytest.fixture
def demo_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the guarded entry points treat the test database as the demo one."""
    monkeypatch.setattr(demo_seed, "DEMO_DATABASE_NAME", "docpilot_test")
    monkeypatch.setenv("DOCPILOT_DEMO", "1")


async def _count_seeded(session: AsyncSession, seeded: SeedSummary) -> int:
    stmt = select(func.count()).select_from(Document).where(Document.id.in_(list(seeded.document_ids.values())))
    return (await session.execute(stmt)).scalar_one()


async def test_seeds_every_document_with_all_the_fields_of_its_schema(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    # 001-025 keep the v1 schema of the committed v1 run; 026-032 are v3, the
    # committed Sonnet extract_v3 run having scored them.
    ids = list(seeded.document_ids.values())
    assert len(ids) == len(artifact["per_doc"]) + len(LABEL_DOC_IDS) == 32
    assert seeded.document_ids == {
        doc_id: demo_document_id(doc_id)
        for doc_id in [e["doc_id"] for e in artifact["per_doc"]] + LABEL_DOC_IDS
    }
    # The committed v1 run keeps its ids: nothing about 001-025 moved.
    assert all(
        seeded.document_ids[e["doc_id"]] == demo_document_id(e["doc_id"]) for e in artifact["per_doc"]
    )

    statuses = (await db_session.execute(select(Document.status).where(Document.id.in_(ids)))).scalars().all()
    assert statuses == ["extracted"] * 32

    rows = (
        await db_session.execute(
            select(Extraction.document_id, Extraction.prompt_version, func.count(ExtractedField.id))
            .join(ExtractedField, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id.in_(ids))
            .group_by(Extraction.document_id, Extraction.prompt_version)
        )
    ).all()
    assert len(rows) == 32
    by_version = {(version, count) for _, version, count in rows}
    assert by_version == {
        (SCHEMA_V1.prompt_version, len(SCHEMA_V1.field_names)),
        (SCHEMA_V3.prompt_version, len(SCHEMA_V3.field_names)),
    }
    assert (len(SCHEMA_V1.field_names), len(SCHEMA_V3.field_names)) == (7, 20)
    assert sum(version == SCHEMA_V3.prompt_version for _, version, _ in rows) == 7


async def test_extraction_replays_the_artifact_with_provenance(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    entry = artifact["per_doc"][5]
    assert entry["doc_id"] != STAGED.doc_id
    extraction = (
        await db_session.execute(
            select(Extraction).where(Extraction.document_id == demo_document_id(entry["doc_id"]))
        )
    ).scalar_one()
    assert extraction.model == artifact["model"]
    assert extraction.prompt_version == artifact["prompt_version"]
    assert float(extraction.cost_usd) == entry["cost_usd"]
    assert extraction.latency_ms == entry["latency_ms"]
    assert (extraction.input_tokens, extraction.output_tokens) == (entry["input_tokens"], entry["output_tokens"])
    # Exactly the 7 model leaves plus the provenance key.
    assert set(extraction.raw_response) == {*SCHEMA_V1.field_names, "_demo_replay"}
    assert extraction.raw_response["_demo_replay"] == {
        "artifact": REPLAY_ARTIFACT.name,
        "doc_id": entry["doc_id"],
        "staged": {},
    }
    for name in SCHEMA_V1.field_names:
        assert extraction.raw_response[name] == {
            "value": entry["fields"][name]["actual"],
            "confidence": entry["fields"][name]["confidence"],
        }
    assert seeded.label_fallbacks == 0
    # Nothing is left to the label: 026-032 replay the committed v3 run, whose
    # recorded cost is part of the replayed total.
    assert seeded.label_replays == 0
    v3_cost = sum(
        e["cost_usd"]
        for e in json.loads(V3_ARTIFACT.read_text(encoding="utf-8"))["per_doc"]
        if e["doc_id"] in LABEL_DOC_IDS
    )
    assert v3_cost == pytest.approx(0.213541)
    assert seeded.total_cost_usd == pytest.approx(artifact["total_cost_usd"] + v3_cost)
    assert seeded.total_cost_usd == pytest.approx(0.525823)


async def test_the_documents_the_v1_run_did_not_score_replay_the_sonnet_v3_run(
    db_session: AsyncSession, seeded: SeedSummary, v3_artifact: dict, v3_entries: dict[str, dict]
) -> None:
    assert (v3_artifact["model"], v3_artifact["prompt_version"]) == ("claude-sonnet-5", "extract_v3")
    # The seed chooses the Sonnet v3 artifact for every one of them, and the
    # Haiku v3 run (a different model, newer by name) is never shown as an answer.
    replays = choose_replays(sorted(seeded.document_ids), json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8")))
    for doc_id in LABEL_DOC_IDS:
        assert replays[doc_id].artifact == V3_ARTIFACT
        assert replays[doc_id].schema is SCHEMA_V3
        assert replays[doc_id].model == "claude-sonnet-5"
    assert HAIKU_V3_ARTIFACT.name > V3_ARTIFACT.name
    assert all(r.artifact != HAIKU_V3_ARTIFACT for r in replays.values())

    extractions = (
        await db_session.execute(
            select(Extraction).where(Extraction.document_id.in_(list(seeded.document_ids.values())))
        )
    ).scalars().all()
    assert {e.model for e in extractions} == {"claude-sonnet-5"}
    by_document = {e.document_id: e for e in extractions}
    for doc_id, entry in v3_entries.items():
        extraction = by_document[seeded.document_ids[doc_id]]
        assert extraction.prompt_version == "extract_v3"
        assert float(extraction.cost_usd) == entry["cost_usd"]
        assert extraction.latency_ms == entry["latency_ms"]
        assert (extraction.input_tokens, extraction.output_tokens) == (
            entry["input_tokens"],
            entry["output_tokens"],
        )
        assert extraction.cache_read_input_tokens == entry["cache_read_input_tokens"]
        provenance = extraction.raw_response["_demo_replay"]
        assert provenance["artifact"] == V3_ARTIFACT.name
        assert provenance["doc_id"] == doc_id
        assert set(extraction.raw_response) == {*SCHEMA_V3.field_names, "_demo_replay"}
        # The model's own answer with its own evidence, not the label's.
        for name in SCHEMA_V3.field_names:
            assert extraction.raw_response[name]["value"] == entry["fields"][name]["actual"]
            assert extraction.raw_response[name]["confidence"] == entry["fields"][name]["confidence"]
            assert extraction.raw_response[name].get("evidence") == entry["fields"][name].get("evidence")


async def test_pending_review_is_the_low_confidence_fields_plus_the_staged_ones(
    db_session: AsyncSession,
    seeded: SeedSummary,
    artifact: dict,
    v3_entries: dict[str, dict],
    settings: Settings,
) -> None:
    staged = {(case.doc_id, case.field_name) for case in STAGED_REVIEW_CASES}
    uncertain_v1 = {
        (entry["doc_id"], name)
        for entry in artifact["per_doc"]
        for name, field in entry["fields"].items()
        if field["confidence"] < settings.review_threshold
    }
    low_v3 = {
        (doc_id, name)
        for doc_id, entry in v3_entries.items()
        for name, field in entry["fields"].items()
        if field["confidence"] < settings.review_threshold
    }
    expected = uncertain_v1 | low_v3 | staged
    # The replayed v1 run itself is uncertain about at least one field, so the
    # expectation is not just the staged cases.
    assert expected - staged
    # The v3 run reports a field it did not read as null at confidence 0.0 (its
    # prompt asks for a low confidence), and the seed stores it as the worker
    # would: below the threshold, so pending. Every low field of the v3 run is
    # such an absent field; none is a value the model read doubtfully.
    assert all(v3_entries[doc_id]["fields"][name]["actual"] in (None, []) for doc_id, name in low_v3)
    assert (len(uncertain_v1), len(low_v3), len(staged)) == (3, 52, 3)

    pending = (
        await db_session.execute(
            select(Extraction.document_id, ExtractedField.field_name)
            .join(ExtractedField, ExtractedField.extraction_id == Extraction.id)
            .where(
                Extraction.document_id.in_(seeded.document_ids.values()),
                ExtractedField.needs_review.is_(True),
                ExtractedField.reviewed_at.is_(None),
            )
        )
    ).all()
    by_demo_id = {demo_id: doc_id for doc_id, demo_id in seeded.document_ids.items()}
    assert {(by_demo_id[document_id], name) for document_id, name in pending} == expected
    assert seeded.pending_review == len(expected) == 58
    assert seeded.staged == len(STAGED_REVIEW_CASES)


async def test_staged_subtotal_for_016_is_pending_and_recorded_in_provenance(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    document_id = seeded.document_ids[STAGED.doc_id]
    field = (
        await db_session.execute(
            select(ExtractedField)
            .join(Extraction, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id == document_id, ExtractedField.field_name == "subtotal")
        )
    ).scalar_one()
    assert field.value == {"value": 46.25, "confidence": 0.68}
    assert field.confidence == 0.68
    assert field.needs_review is True

    replayed = next(e for e in artifact["per_doc"] if e["doc_id"] == STAGED.doc_id)["fields"]["subtotal"]
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document_id))
    ).scalar_one()
    # The model's own leaf is untouched; the override is recorded beside it.
    assert extraction.raw_response["subtotal"] == {
        "value": replayed["actual"],
        "confidence": replayed["confidence"],
    }
    assert extraction.raw_response["_demo_replay"]["staged"] == {
        "subtotal": {
            "replayed": {"value": replayed["actual"], "confidence": replayed["confidence"]},
            "staged": {"value": 46.25, "confidence": 0.68},
            "reason": STAGED.reason,
        }
    }


async def test_nothing_is_spent_today_and_history_is_staggered(
    db_session: AsyncSession, settings: Settings
) -> None:
    before = await spent_today_usd(db_session, NOW)
    seeded = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    assert seeded.total_cost_usd > 0
    assert await spent_today_usd(db_session, NOW) == pytest.approx(before, abs=1e-9)

    ids = list(seeded.document_ids.values())
    today = day_start(NOW)
    for model, id_column in (
        (Document, Document.id),
        (Extraction, Extraction.document_id),
        (DocumentPage, DocumentPage.document_id),
    ):
        stamps = (await db_session.execute(select(model.created_at).where(id_column.in_(ids)))).scalars().all()
        assert stamps
        assert all(today - timedelta(days=HISTORY_DAYS) <= stamp < today for stamp in stamps)

    created = (await db_session.execute(select(Document.created_at).where(Document.id.in_(ids)))).scalars().all()
    assert len(set(created)) == 32
    assert len({stamp.date() for stamp in created}) == HISTORY_DAYS


async def test_images_are_stored_where_the_file_endpoint_serves_them(
    db_session: AsyncSession, seeded: SeedSummary, tmp_path
) -> None:
    documents = (
        await db_session.execute(select(Document).where(Document.id.in_(list(seeded.document_ids.values()))))
    ).scalars().all()
    for document in documents:
        extension = ALLOWED_MIME_TYPES[document.mime_type]
        path = tmp_path / f"{document.id}{extension}"
        assert document.storage_path == str(path)
        assert path.is_file()
        assert document.filename.endswith(extension)


async def test_gold_pages_are_indexed_for_the_documents_that_have_gold_text(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    indexed = dict(
        (
            await db_session.execute(
                select(DocumentPage.document_id, DocumentPage.source).where(
                    DocumentPage.document_id.in_(list(seeded.document_ids.values()))
                )
            )
        ).all()
    )
    # The 25 documents of the retrieval corpus, and none of the v2 documents.
    assert set(indexed) == {demo_document_id(e["doc_id"]) for e in artifact["per_doc"]}
    assert set(indexed.values()) == {"gold"}


async def test_ids_are_the_same_across_a_wipe_and_a_second_seed(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path
) -> None:
    await _wipe(db_session, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 0

    again = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    assert again.document_ids == seeded.document_ids
    assert len(list(tmp_path.iterdir())) == 32


async def test_reset_refuses_the_wrong_database_or_a_missing_env_var(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The env var alone is not enough: this is docpilot_test, not the demo database.
    monkeypatch.setenv("DOCPILOT_DEMO", "1")
    with pytest.raises(RuntimeError, match="docpilot_test"):
        await reset_demo_data(db_session, settings)

    # Right database, but the env var is missing or not "1".
    monkeypatch.setattr(demo_seed, "DEMO_DATABASE_NAME", "docpilot_test")
    monkeypatch.delenv("DOCPILOT_DEMO", raising=False)
    with pytest.raises(RuntimeError, match="refusing"):
        await reset_demo_data(db_session, settings)
    monkeypatch.setenv("DOCPILOT_DEMO", "0")
    with pytest.raises(RuntimeError, match="refusing"):
        await reset_demo_data(db_session, settings)

    assert len(list(tmp_path.iterdir())) == 32
    assert await _count_seeded(db_session, seeded) == 32


@pytest.mark.usefixtures("demo_database")
async def test_reset_deletes_rows_and_leaves_the_files_for_after_the_commit(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path
) -> None:
    old_upload = tmp_path / f"{uuid.uuid4()}.jpg"
    old_upload.write_bytes(b"old upload")

    await reset_demo_data(db_session, settings)
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 0
    assert len(list(tmp_path.iterdir())) == 33

    remove_stale_uploads(settings, seeded)
    assert not old_upload.exists()
    assert len(list(tmp_path.iterdir())) == 32


@pytest.mark.usefixtures("demo_database")
async def test_reset_and_wipe_delete_content_sessions_and_keys_and_never_an_account(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path, second_organization: Organization
) -> None:
    user = await accounts.create_user(db_session, email="ada@example.com", password="correct horse battery")
    for organization_id in (DEFAULT_ORGANIZATION_ID, second_organization.id):
        await accounts.add_membership(
            db_session, organization_id=organization_id, user_id=user.id, role="owner"
        )
    tokens, keys = {}, {}
    for organization_id in (DEFAULT_ORGANIZATION_ID, second_organization.id):
        tokens[organization_id] = await sessions.create_session(
            db_session, user_id=user.id, organization_id=organization_id, ttl=timedelta(hours=1)
        )
        _, keys[organization_id] = await api_keys.create_api_key(
            db_session, organization_id=organization_id, name="demo"
        )

    async def account_rows() -> list[int]:
        counts = []
        for model in (Organization, User, Membership):
            assert model.__tablename__ in IDENTITY_TABLES
            counts.append((await db_session.execute(select(func.count()).select_from(model))).scalar_one())
        return counts

    async def other_organizations_credentials() -> list[int]:
        return [
            (
                await db_session.execute(
                    select(func.count())
                    .select_from(model)
                    .where(model.organization_id != DEFAULT_ORGANIZATION_ID)
                )
            ).scalar_one()
            for model in (AuthSession, ApiKey)
        ]

    accounts_before = await account_rows()
    others_before = await other_organizations_credentials()
    assert all(count >= 1 for count in accounts_before + others_before)

    async def credentials_gone(token: str, key: str) -> bool:
        return (
            await sessions.lookup_session(db_session, token) is None
            and await api_keys.find_active_api_key(db_session, key) is None
        )

    async def second_credentials_kept() -> bool:
        return (
            await sessions.lookup_session(db_session, tokens[second_organization.id]) is not None
            and await api_keys.find_active_api_key(db_session, keys[second_organization.id]) is not None
        )

    await reset_demo_data(db_session, settings)
    assert await _count_seeded(db_session, seeded) == 0
    assert await account_rows() == accounts_before
    assert await credentials_gone(tokens[DEFAULT_ORGANIZATION_ID], keys[DEFAULT_ORGANIZATION_ID])
    assert await other_organizations_credentials() == others_before
    assert await second_credentials_kept()
    assert await db_session.get(Organization, DEFAULT_ORGANIZATION_ID) is not None

    # _wipe deletes the same rows: put the default organization's credentials back first.
    token = await sessions.create_session(
        db_session, user_id=user.id, organization_id=DEFAULT_ORGANIZATION_ID, ttl=timedelta(hours=1)
    )
    _, key = await api_keys.create_api_key(db_session, organization_id=DEFAULT_ORGANIZATION_ID, name="again")
    await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    await _wipe(db_session, tmp_path)
    assert await account_rows() == accounts_before
    assert await credentials_gone(token, key)
    assert await second_credentials_kept()


def test_the_demo_password_is_public_and_fourteen_characters_the_api_accepts() -> None:
    assert len(DEMO_USER_PASSWORD) == 14
    passwords.check_password_length(DEMO_USER_PASSWORD)
    assert accounts.canonical_email(DEMO_USER_EMAIL) == DEMO_USER_EMAIL == "demo@doc-pilot.local"


async def _sign_in(client: AsyncClient, email: str, password: str):
    return await client.post(
        "/auth/login",
        json={"email": email, "password": password},
        headers={"Origin": "http://localhost:3000"},
    )


@pytest.mark.usefixtures("demo_database")
async def test_the_demo_user_exists_owns_the_default_organization_and_signs_in(
    db_session: AsyncSession, anon_client: AsyncClient
) -> None:
    user = await upsert_demo_user(db_session)

    membership = await db_session.get(Membership, (DEFAULT_ORGANIZATION_ID, user.id))
    assert membership is not None and membership.role == MEMBERSHIP_ROLE_OWNER
    assert user.email == DEMO_USER_EMAIL and user.disabled_at is None

    response = await _sign_in(anon_client, DEMO_USER_EMAIL, DEMO_USER_PASSWORD)
    assert response.status_code == 200
    assert response.json()["organization"]["role"] == MEMBERSHIP_ROLE_OWNER
    assert (await anon_client.get("/documents")).status_code == 200
    assert (await _sign_in(anon_client, DEMO_USER_EMAIL, "not the password")).status_code == 401


@pytest.mark.usefixtures("demo_database")
async def test_upserting_again_resets_a_changed_demo_user_and_ends_its_sessions(
    db_session: AsyncSession, anon_client: AsyncClient, second_organization: Organization
) -> None:
    user = await upsert_demo_user(db_session)
    user_id, second_id = user.id, second_organization.id
    await accounts.set_password(db_session, user, "a password someone chose")
    user.disabled_at = datetime.now(UTC)
    membership = await db_session.get(Membership, (DEFAULT_ORGANIZATION_ID, user_id))
    membership.role = MEMBERSHIP_ROLE_MEMBER
    await accounts.add_membership(
        db_session, organization_id=second_id, user_id=user_id, role=MEMBERSHIP_ROLE_MEMBER
    )
    token = await sessions.create_session(
        db_session, user_id=user_id, organization_id=DEFAULT_ORGANIZATION_ID, ttl=timedelta(hours=1)
    )
    await db_session.flush()
    assert (await _sign_in(anon_client, DEMO_USER_EMAIL, DEMO_USER_PASSWORD)).status_code == 401

    again = await upsert_demo_user(db_session)

    assert again.id == user_id
    assert (
        await db_session.execute(select(func.count()).select_from(User).where(User.email == DEMO_USER_EMAIL))
    ).scalar_one() == 1
    assert again.disabled_at is None
    assert await sessions.lookup_session(db_session, token) is None
    assert (await _sign_in(anon_client, DEMO_USER_EMAIL, DEMO_USER_PASSWORD)).status_code == 200
    db_session.expire_all()
    assert (await db_session.get(Membership, (DEFAULT_ORGANIZATION_ID, user_id))).role == MEMBERSHIP_ROLE_OWNER
    # Only the default organization's role is the demo's to set.
    other = await db_session.get(Membership, (second_id, user_id))
    assert other.role == MEMBERSHIP_ROLE_MEMBER


async def test_the_demo_user_is_only_ever_made_on_the_demo_database(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def demo_users() -> int:
        stmt = select(func.count()).select_from(User).where(User.email == DEMO_USER_EMAIL)
        return (await db_session.execute(stmt)).scalar_one()

    before = await demo_users()
    # The env var alone is not enough: this is docpilot_test, not the demo database.
    monkeypatch.setenv("DOCPILOT_DEMO", "1")
    with pytest.raises(RuntimeError, match="docpilot_test"):
        await upsert_demo_user(db_session)

    # Right database, but the env var is missing or not "1".
    monkeypatch.setattr(demo_seed, "DEMO_DATABASE_NAME", "docpilot_test")
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv("DOCPILOT_DEMO", raising=False)
        else:
            monkeypatch.setenv("DOCPILOT_DEMO", value)
        with pytest.raises(RuntimeError, match="refusing"):
            await upsert_demo_user(db_session)
    assert await demo_users() == before


@pytest.mark.usefixtures("demo_database")
async def test_reset_deletes_only_the_bound_organizations_content(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, second_organization: Organization
) -> None:
    await bind_tenant(db_session, second_organization.id)
    other = Document(filename="other.png", mime_type="image/png", storage_path="/tmp/o", status="extracted")
    db_session.add(other)
    await db_session.flush()
    other_id = other.id
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)

    await reset_demo_data(db_session, settings)

    assert await _count_seeded(db_session, seeded) == 0
    await bind_tenant(db_session, second_organization.id)
    assert (await db_session.execute(select(Document.id).where(Document.id == other_id))).scalar_one() == other_id


@pytest.mark.usefixtures("demo_database")
@pytest.mark.parametrize("foreign", ["notes.txt", "model.onnx", f"{uuid.uuid4()}.exe", "not-a-uuid.png", "subdir"])
async def test_an_upload_dir_holding_anything_but_uploads_is_never_touched(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path, foreign: str
) -> None:
    if foreign == "subdir":
        (tmp_path / foreign).mkdir()
    else:
        (tmp_path / foreign).write_text("not an upload")

    with pytest.raises(RuntimeError, match="not uploads"):
        await reset_demo_data(db_session, settings)
    with pytest.raises(RuntimeError, match="not uploads"):
        remove_stale_uploads(settings, seeded)
    with pytest.raises(RuntimeError, match="not uploads"):
        await _wipe(db_session, tmp_path)
    with pytest.raises(RuntimeError, match="not uploads"):
        await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)

    assert len(list(tmp_path.iterdir())) == 33
    assert await _count_seeded(db_session, seeded) == 32


def _load_cli():
    spec = importlib.util.spec_from_file_location("seed_demo_cli", BACKEND_DIR / "scripts" / "seed_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_cli_refuses_a_database_that_is_not_the_demo_one(
    db_session: AsyncSession, settings: Settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCPILOT_DEMO", "1")
    cli = _load_cli()
    for reset in (False, True):
        with pytest.raises(RuntimeError, match="docpilot_test"):
            await cli.run(db_session, settings, reset=reset)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("demo_database")
async def test_cli_refuses_a_non_empty_database_without_reset_and_replaces_it_with_reset(
    db_session: AsyncSession, settings: Settings, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    db_session.add(Document(filename="x.png", mime_type="image/png", storage_path="/x", status="uploaded"))
    await db_session.flush()

    assert await cli.run(db_session, settings, reset=False) == 1
    assert "--reset" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []

    assert await cli.run(db_session, settings, reset=True) == 0
    out = capsys.readouterr().out
    assert "Seeded 32 documents, 58 fields pending review (3 staged)" in out
    assert "replayed cost $0.5258" in out
    assert "0 label fallbacks, 0 label replays" in out
    assert f"Demo sign-in: {DEMO_USER_EMAIL} / {DEMO_USER_PASSWORD}" in out
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 32
    assert len(list(tmp_path.iterdir())) == 32
    user = await accounts.get_user(db_session, DEMO_USER_EMAIL)
    assert (await db_session.get(Membership, (DEFAULT_ORGANIZATION_ID, user.id))).role == MEMBERSHIP_ROLE_OWNER

    # Resetting again keeps the one user and still prints the sign-in.
    assert await cli.run(db_session, settings, reset=True) == 0
    assert "Demo sign-in:" in capsys.readouterr().out
    assert (await accounts.get_user(db_session, DEMO_USER_EMAIL)).id == user.id


async def test_seed_reconciles_every_extraction_and_only_the_staged_misreads_fail(
    db_session: AsyncSession, seeded: SeedSummary
) -> None:
    ids = list(seeded.document_ids.values())
    rows = (
        await db_session.execute(
            select(Extraction.document_id, Reconciliation.status)
            .join(Reconciliation, Reconciliation.extraction_id == Extraction.id)
            .where(Extraction.document_id.in_(ids))
        )
    ).all()
    statuses = {document_id: status for document_id, status in rows}
    assert set(statuses) == set(ids)
    assert {d for d, status in statuses.items() if status == "fail"} == {
        seeded.document_ids[STAGED.doc_id],
        seeded.document_ids[QUOTE_STAGED[0].doc_id],
    }
    assert set(statuses.values()) <= {"pass", "fail"}


async def _flagged(db_session: AsyncSession, document_id: uuid.UUID) -> dict[str, str]:
    fields = (
        await db_session.execute(
            select(ExtractedField)
            .join(Extraction, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id == document_id)
        )
    ).scalars()
    return {f.field_name: f.review_reason for f in fields if f.needs_review}


async def test_the_staged_subtotal_is_flagged_by_the_arithmetic_and_only_confidence_flags_the_rest(
    db_session: AsyncSession, seeded: SeedSummary
) -> None:
    # Only the subtotal: the suggestion names it, so the line items, tax, and
    # total that its misread contradicts are not queued.
    assert await _flagged(db_session, seeded.document_ids[STAGED.doc_id]) == {
        "subtotal": "reconciliation:lines_sum"
    }
    # The three uncertain fields of the replayed v1 run, the 52 fields the v3
    # run left null at confidence 0.0, the staged subtotal, and the two staged
    # amounts of 028 (see the pending-review test for the breakdown).
    assert seeded.pending_review == 58

    every_field = (
        await db_session.execute(
            select(ExtractedField)
            .join(Extraction, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id.in_(list(seeded.document_ids.values())))
        )
    ).scalars()
    for field in every_field:
        assert field.needs_review == (field.review_reason is not None)
        if field.needs_review and field.field_name not in {"subtotal", "amount_paid", "amount_due"}:
            assert field.review_reason == "confidence"


async def test_a_document_no_run_scored_replays_its_label_with_gold_evidence(
    db_session: AsyncSession, label_seeded: SeedSummary
) -> None:
    seeded = label_seeded
    # No later run in the (empty) results directory: all seven are label replays,
    # and the queue holds the v1 run's three uncertain fields and the three staged.
    assert seeded.label_replays == 7
    assert seeded.pending_review == 6
    doc_id = "026-invoice-discount-shipping-two-taxes"
    case = next(c for c in load_cases() if c.doc_id == doc_id)
    extraction = (
        await db_session.execute(
            select(Extraction).where(Extraction.document_id == seeded.document_ids[doc_id])
        )
    ).scalar_one()
    assert (extraction.model, extraction.prompt_version) == (LABEL_REPLAY_MODEL, "extract_v2")
    assert (extraction.cost_usd, extraction.latency_ms) == (0, 0)
    assert (extraction.input_tokens, extraction.output_tokens) == (0, 0)
    assert extraction.raw_response["_demo_replay"] == {"source": "label", "doc_id": doc_id, "staged": {}}
    # Exactly the label as a fully confident answer with its gold quotes.
    expected = label_tool_input(case.fields, evidence=case.evidence)
    assert {k: v for k, v in extraction.raw_response.items() if k != "_demo_replay"} == expected
    assert expected["tax"]["evidence"] == case.evidence["tax"]

    fields = {
        f.field_name: f
        for f in (
            await db_session.execute(select(ExtractedField).where(ExtractedField.extraction_id == extraction.id))
        ).scalars()
    }
    assert fields["discount"].value == {
        "value": 23.92,
        "confidence": 1.0,
        "evidence": case.evidence["discount"],
    }
    assert [row["amount"]["value"] for row in fields["tax_lines"].value] == [29.54, 5.68]
    assert not any(f.needs_review for f in fields.values())


async def test_the_staged_028_amounts_pass_the_arithmetic_and_fail_their_quotes(
    db_session: AsyncSession, seeded: SeedSummary, v3_entries: dict[str, dict]
) -> None:
    document_id = seeded.document_ids[QUOTE_STAGED[0].doc_id]
    case = next(c for c in load_cases() if c.doc_id == QUOTE_STAGED[0].doc_id)
    # The staged leaf keeps the quote the committed v3 run cited, and that quote
    # prints the true amount (here the gold one), so the staged value disagrees.
    for staged_case in QUOTE_STAGED:
        replayed = v3_entries[staged_case.doc_id]["fields"][staged_case.field_name]
        assert replayed["actual"] == case.fields[staged_case.field_name]
        assert replayed["evidence"] == case.evidence[staged_case.field_name]
    assert {c.field_name for c in QUOTE_STAGED} == {"amount_paid", "amount_due"}
    # The misreads are the printed amounts moved by one number in opposite
    # directions, so total - paid = due still holds.
    shifted = {c.field_name: c.value for c in QUOTE_STAGED}
    assert case.fields["total"] - shifted["amount_paid"] == pytest.approx(shifted["amount_due"])
    assert shifted["amount_paid"] != case.fields["amount_paid"]

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document_id))
    ).scalar_one()
    for staged_case in QUOTE_STAGED:
        name = staged_case.field_name
        leaf = {
            "value": staged_case.value,
            "confidence": staged_case.confidence,
            "evidence": case.evidence[name],
        }
        field = (
            await db_session.execute(
                select(ExtractedField).where(
                    ExtractedField.extraction_id == extraction.id, ExtractedField.field_name == name
                )
            )
        ).scalar_one()
        # The quote is the one printed, so it is what disagrees with the value.
        assert field.value == leaf
        assert field.review_reason == f"quote:{name}"
        provenance = extraction.raw_response["_demo_replay"]["staged"][name]
        assert provenance["staged"] == leaf
        assert provenance["replayed"]["value"] == case.fields[name]
        assert provenance["reason"] == staged_case.reason

    report = (
        await db_session.execute(select(Reconciliation.report).where(Reconciliation.extraction_id == extraction.id))
    ).scalar_one()
    status = {check["id"]: check["status"] for check in report["checks"]}
    assert status["balance_math"] == "pass"
    assert {i for i, st in status.items() if st == "fail"} == {"quote:amount_paid", "quote:amount_due"}
    assert report["suggestion"] is None


def _v2_artifact(
    model: str, doc_ids: list[str], started: str, prompt_version: str = "extract_v2"
) -> dict:
    """A scored artifact of an evidence-carrying prompt version (extract_v2 unless
    told otherwise) holding `doc_ids`, answered from their labels at confidence
    0.9, in the shape evals/run.py writes."""
    cases = {c.doc_id: c for c in load_cases()}
    per_doc = []
    for doc_id in doc_ids:
        answers = label_tool_input(cases[doc_id].fields, confidence=0.9, evidence=cases[doc_id].evidence)
        fields = {}
        for name, leaf in answers.items():
            fields[name] = {"actual": leaf["value"], "confidence": leaf["confidence"]}
            if leaf.get("evidence") is not None:
                fields[name]["evidence"] = leaf["evidence"]
        per_doc.append(
            {
                "doc_id": doc_id,
                "status": "scored",
                "fields": fields,
                "cost_usd": 0.05,
                "latency_ms": 9000,
                "input_tokens": 3000,
                "output_tokens": 1500,
            }
        )
    return {
        "model": model,
        "prompt_version": prompt_version,
        "started_at_utc": started,
        "per_doc": per_doc,
    }


def test_a_later_run_replays_instead_of_the_label_and_only_for_the_same_model(
    artifact: dict, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_ids = sorted([e["doc_id"] for e in artifact["per_doc"]] + LABEL_DOC_IDS)
    monkeypatch.setattr(demo_seed, "RESULTS_DIR", tmp_path)

    only_labels = choose_replays(doc_ids, artifact)
    assert {d for d, r in only_labels.items() if r.entry is None} == set(LABEL_DOC_IDS)
    assert all(r.schema is SCHEMA_V1 for d, r in only_labels.items() if d not in LABEL_DOC_IDS)

    older = _v2_artifact(artifact["model"], LABEL_DOC_IDS[:3] + LABEL_DOC_IDS[-1:], "old")
    newer = _v2_artifact(artifact["model"], LABEL_DOC_IDS[:2], "new")
    haiku = _v2_artifact("claude-haiku-4-5", LABEL_DOC_IDS, "newest")
    for name, run in (
        ("20260901T000000Z_claude-sonnet-5_extract_v2.json", older),
        ("20260902T000000Z_claude-sonnet-5_extract_v2.json", newer),
        ("20260903T000000Z_claude-haiku-4-5_extract_v2.json", haiku),
        # A v1 run is never a later run, whatever its name says.
        ("20260904T000000Z_claude-sonnet-5_extract_v1.json", artifact),
    ):
        (tmp_path / name).write_text(json.dumps(run), encoding="utf-8")

    replays = choose_replays(doc_ids, artifact)
    for doc_id in LABEL_DOC_IDS[:2]:
        assert replays[doc_id].artifact.name == "20260902T000000Z_claude-sonnet-5_extract_v2.json"
    for doc_id in LABEL_DOC_IDS[2:3] + LABEL_DOC_IDS[-1:]:
        assert replays[doc_id].artifact.name == "20260901T000000Z_claude-sonnet-5_extract_v2.json"
    for doc_id in LABEL_DOC_IDS[3:-1]:
        assert replays[doc_id].entry is None
    assert all(r.model == artifact["model"] for d, r in replays.items() if r.entry is not None)
    # 001-025 still replay the committed v1 run.
    assert all(replays[e["doc_id"]].artifact == REPLAY_ARTIFACT for e in artifact["per_doc"])


def test_a_v3_run_is_a_later_run_too_and_the_newest_one_wins(
    artifact: dict, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_ids = sorted([e["doc_id"] for e in artifact["per_doc"]] + LABEL_DOC_IDS)
    monkeypatch.setattr(demo_seed, "RESULTS_DIR", tmp_path)
    v2 = _v2_artifact(artifact["model"], LABEL_DOC_IDS[:4], "v2")
    v3 = _v2_artifact(artifact["model"], LABEL_DOC_IDS[:2], "v3", prompt_version="extract_v3")
    unknown = _v2_artifact(artifact["model"], LABEL_DOC_IDS, "later", prompt_version="extract_v9")
    for name, run in (
        ("20260901T000000Z_claude-sonnet-5_extract_v2.json", v2),
        ("20260902T000000Z_claude-sonnet-5_extract_v3.json", v3),
        # A prompt version this code does not know is never replayed.
        ("20260903T000000Z_claude-sonnet-5_extract_v9.json", unknown),
    ):
        (tmp_path / name).write_text(json.dumps(run), encoding="utf-8")

    replays = choose_replays(doc_ids, artifact)

    for doc_id in LABEL_DOC_IDS[:2]:
        assert replays[doc_id].artifact.name == "20260902T000000Z_claude-sonnet-5_extract_v3.json"
        assert replays[doc_id].schema is SCHEMA_V3
    for doc_id in LABEL_DOC_IDS[2:4]:
        assert replays[doc_id].artifact.name == "20260901T000000Z_claude-sonnet-5_extract_v2.json"
        assert replays[doc_id].schema is SCHEMA_V2
    for doc_id in LABEL_DOC_IDS[4:]:
        assert replays[doc_id].entry is None


async def test_a_v3_run_is_seeded_under_its_own_prompt_version(
    db_session: AsyncSession, settings: Settings, artifact: dict, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path.parent / f"{tmp_path.name}-v3-runs"
    runs.mkdir()
    doc_id = LABEL_DOC_IDS[0]
    name = "20260901T000000Z_claude-sonnet-5_extract_v3.json"
    run = _v2_artifact(artifact["model"], [doc_id], "x", prompt_version="extract_v3")
    (runs / name).write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(demo_seed, "RESULTS_DIR", runs)

    seeded = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == seeded.document_ids[doc_id]))
    ).scalar_one()
    assert (extraction.model, extraction.prompt_version) == (artifact["model"], "extract_v3")
    assert extraction.raw_response["_demo_replay"]["artifact"] == name
    assert extraction.raw_response["total"]["evidence"]["quote"]


async def test_a_later_run_is_seeded_with_its_provenance_and_evidence(
    db_session: AsyncSession, settings: Settings, artifact: dict, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path.parent / f"{tmp_path.name}-runs"
    runs.mkdir()
    doc_id = LABEL_DOC_IDS[0]
    name = "20260901T000000Z_claude-sonnet-5_extract_v2.json"
    (runs / name).write_text(json.dumps(_v2_artifact(artifact["model"], [doc_id], "x")), encoding="utf-8")
    monkeypatch.setattr(demo_seed, "RESULTS_DIR", runs)

    seeded = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    assert seeded.label_replays == 6
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == seeded.document_ids[doc_id]))
    ).scalar_one()
    assert (extraction.model, extraction.prompt_version) == (artifact["model"], "extract_v2")
    assert (float(extraction.cost_usd), extraction.latency_ms) == (0.05, 9000)
    assert extraction.raw_response["_demo_replay"] == {"artifact": name, "doc_id": doc_id, "staged": {}}
    assert extraction.raw_response["total"]["confidence"] == 0.9
    assert extraction.raw_response["total"]["evidence"]["quote"]
