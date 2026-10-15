"""Which fields a person has to look at: model doubt and failed arithmetic.

A field is flagged `needs_review` for one reason, stored in `review_reason`:
`confidence` (the model was unsure of it), `reconciliation:<check_id>` (a
failing arithmetic check in app/reconcile.py names it), or `quote:<path>` (the
number in the field's evidence quote is not its value; the check's id is
already that text). `route` is the pure rule that
turns a field set plus a reconciliation `Report` into those flags, and
`refresh_reconciliation` is the one place that runs it against the database:
the worker, the demo seed, the eval corpus, every review resolve, and
scripts/reconcile_all.py all call it, so the stored flags never drift from the
stored report.

Routing, from scratch on every call (so a correction can clear what it fixed):

1. A reviewed field (`reviewed_at` set) is frozen: its flag and reason keep
   whatever they were when a person resolved it. A person's word is never
   re-litigated by a later recomputation.
2. For each failing check, quote checks first and then in report order, the
   suspects are the field the
   report's suggestion targets, when there is one and nobody has reviewed it;
   otherwise every unreviewed field the check involves. An operand path such
   as `line_items[2].total` belongs to the `line_items` field, and
   `tax_lines[1].amount` to `tax_lines`. The suggestion
   is the one change that fixes every failing check, so it is the likeliest
   misread and the other operands are not worth a person's time yet. After a
   person approves a value that still fails, no suggestion can target it, and
   the remaining operands are flagged instead.
3. An unreviewed suspect is flagged with the first failing check that names it,
   so a field blamed by both a quote check and an arithmetic check carries the
   quote reason: it says what is wrong with that one value, where the
   arithmetic only says the numbers do not add up. Any other unreviewed field is flagged `confidence` when its confidence is
   under the review threshold, and is not flagged otherwise.

`needs_review` is true exactly when `review_reason` is set (a CHECK enforces it).

Concurrency: `lock_extraction` takes the extraction row's lock. Whoever
changes a field's review state locks first, then changes it, then calls
`refresh_reconciliation`; otherwise two reviewers resolving different fields of
one document could each wait on a field row the other holds.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    REVIEW_REASON_CONFIDENCE,
    ExtractedField,
    Extraction,
    Reconciliation,
    reconciliation_reason,
)
from app.reconcile import (
    KIND_QUOTE,
    Check,
    ReconcileInput,
    Report,
    reconcile,
    report_to_json,
)

_FIELD_NAME_RE = re.compile(r"[^.\[]+")


def field_name_of(path: str) -> str:
    """The field an operand path belongs to: `line_items[2].total` is the
    `line_items` field, `tax_lines[1].amount` is the `tax_lines` field,
    `subtotal` is `subtotal`."""
    match = _FIELD_NAME_RE.match(path)
    return match.group() if match else path


@dataclass(frozen=True)
class Routing:
    """A field's review flag and the reason for it; `review_reason` is None
    exactly when `needs_review` is False."""

    needs_review: bool
    review_reason: str | None


NOT_FLAGGED = Routing(False, None)


def _reason(check: Check) -> str:
    """The review reason a failing check gives the fields it flags: a quote
    check's id is already `quote:<path>`, an arithmetic check's is prefixed."""
    return check.id if check.kind == KIND_QUOTE else reconciliation_reason(check.id)


def route(
    fields: Iterable[ExtractedField], report: Report, review_threshold: float
) -> dict[str, Routing]:
    """The routing of every field, keyed by field name. Pure: nothing is
    changed, and a reviewed field maps to the flag it already has."""
    fields = list(fields)
    reviewed = {f.field_name for f in fields if f.reviewed_at is not None}
    names = {f.field_name for f in fields}
    suggested = field_name_of(report.suggestion.path) if report.suggestion else None
    if suggested is not None and (suggested in reviewed or suggested not in names):
        suggested = None

    # A stable sort puts the quote checks first, so the reason a field gets is
    # the quote's whenever one blames it.
    failing = sorted(report.failing_checks, key=lambda check: check.kind != KIND_QUOTE)
    flagged: dict[str, Routing] = {}
    if suggested is not None and failing:
        # The suggestion fixes every failing check; the reason is the first
        # one that names the field (the first of all, if none does).
        culprit = next(
            (c for c in failing if suggested in {field_name_of(p) for p in c.fields}), failing[0]
        )
        flagged[suggested] = Routing(True, _reason(culprit))
    else:
        for check in failing:
            for name in dict.fromkeys(field_name_of(path) for path in check.fields):
                if name in names and name not in reviewed:
                    flagged.setdefault(name, Routing(True, _reason(check)))

    routed: dict[str, Routing] = {}
    for field in fields:
        if field.reviewed_at is not None:
            routed[field.field_name] = Routing(field.needs_review, field.review_reason)
        elif field.field_name in flagged:
            routed[field.field_name] = flagged[field.field_name]
        elif field.confidence < review_threshold:
            routed[field.field_name] = Routing(True, REVIEW_REASON_CONFIDENCE)
        else:
            routed[field.field_name] = NOT_FLAGGED
    return routed


@dataclass(frozen=True)
class FieldChange:
    field_name: str
    before: Routing
    after: Routing


@dataclass(frozen=True)
class Refresh:
    """What `refresh_reconciliation` did: the stored row, the report behind
    it, whether the row is new, and every field whose flag or reason changed."""

    reconciliation: Reconciliation
    report: Report
    created: bool
    changes: tuple[FieldChange, ...]


async def lock_extraction(session: AsyncSession, extraction_id: Any) -> None:
    """Take the extraction row's lock for the rest of the transaction."""
    stmt = select(Extraction.id).where(Extraction.id == extraction_id).with_for_update()
    (await session.execute(stmt)).scalar_one()


async def refresh_reconciliation(
    session: AsyncSession, extraction_id: Any, settings: Settings
) -> Refresh:
    """Recompute one extraction's reconciliation and routing and store both in
    the caller's transaction.

    Locks the extraction row, reads its fields (a reviewer's correction wins,
    through `ReconcileInput.from_fields`), reconciles, routes, updates the
    fields whose flag changed, and upserts the `reconciliations` row. It
    flushes nothing it does not need and never commits: the caller owns the
    transaction. The fields are read fresh from the database after the lock, so
    a decision committed by someone else a moment ago is seen.
    """
    await lock_extraction(session, extraction_id)
    stmt = (
        select(ExtractedField)
        .where(ExtractedField.extraction_id == extraction_id)
        .order_by(ExtractedField.field_name)
        .execution_options(populate_existing=True)
    )
    fields = list((await session.execute(stmt)).scalars())
    report = reconcile(ReconcileInput.from_fields(fields))
    routed = route(fields, report, settings.review_threshold)

    changes = []
    for field in fields:
        after = routed[field.field_name]
        before = Routing(field.needs_review, field.review_reason)
        if after != before:
            field.needs_review = after.needs_review
            field.review_reason = after.review_reason
            changes.append(FieldChange(field.field_name, before, after))

    existing = (
        await session.execute(
            select(Reconciliation)
            .where(Reconciliation.extraction_id == extraction_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    row = existing or Reconciliation(extraction_id=extraction_id)
    row.rules_version = report.rules_version
    row.status = report.status
    row.report = report_to_json(report)
    row.computed_at = datetime.now(UTC)
    if existing is None:
        session.add(row)
    await session.flush()
    return Refresh(row, report, existing is None, tuple(changes))


def failing_checks_for(report: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    """The failing checks of a stored report (`report_to_json`) that involve
    one field, in report order."""
    return [
        check
        for check in report.get("checks", ())
        if check["status"] == "fail"
        and any(field_name_of(path) == field_name for path in check["fields"])
    ]


def suggestion_for(report: dict[str, Any], field_name: str) -> dict[str, Any] | None:
    """The stored report's suggestion, when it targets this field."""
    suggestion = report.get("suggestion")
    if suggestion is not None and field_name_of(suggestion["path"]) == field_name:
        return suggestion
    return None
