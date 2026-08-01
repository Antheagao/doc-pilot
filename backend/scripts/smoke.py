"""Live smoke test for the T5 extraction pipeline.

Generates a small synthetic receipt image with PIL, inserts a Document +
Job row directly against the real database (bypassing the upload
endpoint), runs exactly one worker cycle (`app.worker.run_once`) against
the *real* Anthropic API using the real extraction handler
(`app.extraction.process_document_job`), prints the extracted fields,
confidences, token usage, cost, and latency, then deletes the rows and
temp file it created.

This is a manual, opt-in script -- it is not part of the pytest suite
(no network calls or API spend in `pytest`). Run it from `backend/` with
the venv active and `ANTHROPIC_API_KEY` set (via backend/.env, already
gitignored):

    cd backend
    python scripts/smoke.py

See tests/test_live_smoke.py for the assertion-bearing counterpart to
this script: same pipeline, opt-in via RUN_LIVE_SMOKE=1 and pytest's
`live` marker, but with pass/fail assertions instead of a human-readable
console dump.

Requires pillow (dev dependency, see pyproject.toml `[project.optional-dependencies].dev`).
"""

import asyncio
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import delete, select

from app.db import async_session_maker, engine
from app.models import Document, ExtractedField, Extraction, Job
from app.worker import run_once


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Prefer a real TTF for legibility; fall back to PIL's bitmap default
    font if none is found (keeps the script portable off Windows)."""
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_receipt_image(path: Path) -> None:
    """Draw a tiny synthetic receipt: vendor, date, one line item,
    subtotal/tax/total, and a currency symbol -- enough surface for the
    model to populate every field in the extraction schema.
    """
    width, height = 420, 360
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    bold_font = _load_font(20)

    lines = [
        ("Doc-Pilot Coffee Roasters", bold_font),
        ("123 Roast St, Seattle WA", font),
        ("", font),
        ("Date: 2026-03-14", font),
        ("", font),
        ("Item              Qty  Price   Total", font),
        ("House Blend 12oz    2  $14.00  $28.00", font),
        ("", font),
        ("Subtotal:              $28.00", font),
        ("Tax:                    $2.52", font),
        ("Total:                 $30.52", font),
        ("", font),
        ("Currency: USD", font),
    ]

    y = 20
    for text, line_font in lines:
        draw.text((20, y), text, fill="black", font=line_font)
        y += 24

    image.save(path)


async def insert_document_and_job(image_path: Path) -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session_maker() as session:
        document = Document(
            filename="smoke_receipt.png",
            mime_type="image/png",
            storage_path=str(image_path),
            status="uploaded",
        )
        session.add(document)
        await session.flush()
        job = Job(document_id=document.id, state="pending")
        session.add(job)
        await session.commit()
        return document.id, job.id


async def print_results(document_id: uuid.UUID) -> None:
    async with async_session_maker() as session:
        document = await session.get(Document, document_id)
        print(f"\nDocument status: {document.status}")

        extraction = (
            (
                await session.execute(
                    select(Extraction)
                    .where(Extraction.document_id == document_id)
                    .order_by(Extraction.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        if extraction is None:
            print("No extraction row was created -- job likely failed. Check job.last_error.")
            return

        print(f"Model:        {extraction.model}")
        print(f"Prompt ver:   {extraction.prompt_version}")
        print(f"Input tokens: {extraction.input_tokens}")
        print(f"Output tokens:{extraction.output_tokens}")
        print(f"Cost (USD):   ${extraction.cost_usd:.6f}")
        print(f"Latency (ms): {extraction.latency_ms}")

        fields = (
            (
                await session.execute(
                    select(ExtractedField).where(
                        ExtractedField.extraction_id == extraction.id
                    )
                )
            )
            .scalars()
            .all()
        )
        print("\nExtracted fields:")
        for field in sorted(fields, key=lambda f: f.field_name):
            flag = " <-- needs review" if field.needs_review else ""
            print(
                f"  {field.field_name:15s} confidence={field.confidence:.2f}"
                f" value={field.value}{flag}"
            )


async def cleanup(document_id: uuid.UUID, job_id: uuid.UUID) -> None:
    async with async_session_maker() as session:
        extraction_ids = select(Extraction.id).where(Extraction.document_id == document_id)
        await session.execute(
            delete(ExtractedField).where(ExtractedField.extraction_id.in_(extraction_ids))
        )
        await session.execute(delete(Extraction).where(Extraction.document_id == document_id))
        await session.execute(delete(Job).where(Job.id == job_id))
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "smoke_receipt.png"
        generate_receipt_image(image_path)
        print(f"Generated synthetic receipt: {image_path}")

        document_id, job_id = await insert_document_and_job(image_path)
        print(f"Inserted document={document_id} job={job_id}")

        print("Running one worker cycle against the real Anthropic API...")
        claimed = await run_once()
        if not claimed:
            print("run_once claimed no job -- something is wrong with the insert above.")

        await print_results(document_id)
        await cleanup(document_id, job_id)
        print("\nCleaned up document/job/extraction rows.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
