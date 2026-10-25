"""Replaying what a committed extraction run returned.

The committed runs in evals/results/ hold, per document, the answer a real model
gave (each field's value, confidence, and for a schema with evidence its quote)
and what the call cost in tokens and time. Two stand-ins replay them instead of
calling a model: the demo seed (app/demo_seed.py), which writes the rows
directly, and the scripted model (app/scripted_model.py), which answers a
request the worker's real code made. Both choose an answer the same way, so this
module is their one home: the primary run for a document it scored, else the
newest run of the same model under a schema with evidence, else the label.

Nothing here touches a database or a model; it reads the artifacts and the
labels.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.evals.checks import label_tool_input
from app.evals.dataset import EVALS_DIR, EvalCase
from app.extraction_fields import SCHEMA_V2, SCHEMAS, ExtractionSchema, schema_for

RESULTS_DIR = EVALS_DIR / "results"
REPLAY_ARTIFACT = RESULTS_DIR / "20260731T064853Z_claude-sonnet-5_extract_v1.json"
# The committed run whose per-document latency and tokens stand in for
# transcription, which no run measured: a vision model (Haiku, the transcription
# model's default) reading the same pictures. A proxy, and labeled as one wherever
# the numbers are published (app/scripted_model.py).
TRANSCRIPTION_PROXY_ARTIFACT = RESULTS_DIR / "20260731T065007Z_claude-haiku-4-5_extract_v1.json"

# What a document that no run scored shows as its model: its label, replayed.
LABEL_REPLAY_MODEL = "gold-label"
LABEL_REPLAY_PROMPT_VERSION = SCHEMA_V2.prompt_version


def _label_tool_input(case: EvalCase, schema: ExtractionSchema) -> dict:
    """The label as a tool_input at confidence 1.0, shaped like
    seed_labeled_corpus's fields: a schema with evidence also gets the label's
    gold evidence on every leaf."""
    full = label_tool_input(case.fields, evidence=case.evidence if schema.evidence else None)
    return {name: full[name] for name in schema.field_names}


def _replayed_tool_input(
    entry: dict, case: EvalCase, schema: ExtractionSchema
) -> tuple[dict, bool]:
    """Rebuild the tool_input the model returned from an artifact per_doc
    entry: each field's `actual` and `confidence`, and for a schema with
    evidence the evidence the entry kept for a header field (the cells of an
    array field keep theirs inside `actual`; a leaf with none replays as
    evidence null, as the worker stores it). Returns (tool_input,
    used_label_fallback). An entry with no `actual` key at all (an older
    artifact shape) falls back to the label at confidence 1.0 for the whole
    document; the committed Sonnet run has none, and the seed counts them."""
    fields = entry["fields"]
    if any("actual" not in fields.get(name, {}) for name in schema.field_names):
        return _label_tool_input(case, schema), True
    tool_input = {}
    for name in schema.field_names:
        leaf = {"value": fields[name]["actual"], "confidence": fields[name]["confidence"]}
        if schema.evidence:
            leaf["evidence"] = fields[name].get("evidence")
        tool_input[name] = leaf
    return tool_input, False


@dataclass(frozen=True)
class Replay:
    """Where one document's replayed extraction comes from: a committed
    artifact and its scored entry, or (`artifact` and `entry` None) the label."""

    schema: ExtractionSchema
    model: str
    artifact: Path | None = None
    entry: dict | None = None


def _scored_entries(artifact: dict) -> dict[str, dict]:
    return {e["doc_id"]: e for e in artifact["per_doc"] if e["status"] == "scored"}


def _evidence_artifacts(
    model: str, results_dir: Path = RESULTS_DIR
) -> list[tuple[Path, dict]]:
    """The committed runs of `model` in `results_dir` whose schema carries
    evidence (extract_v2, extract_v3, ...), newest first (the file names start
    with their UTC start time). A run of another model is not shown as this
    one's answer, and neither is one of a prompt version this code does not
    know."""
    runs = []
    for path in sorted(results_dir.glob("*_extract_v*.json"), reverse=True):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        schema = SCHEMAS.get(artifact["prompt_version"])
        if artifact["model"] == model and schema is not None and schema.evidence:
            runs.append((path, artifact))
    return runs


def choose_replays(
    doc_ids: list[str], primary: dict, results_dir: Path = RESULTS_DIR
) -> dict[str, Replay]:
    """The replay of each document: the primary artifact's entry when it scored
    the document (001-025, unchanged), else the newest committed run of the same
    model that did under a schema with evidence (any such prompt version), else
    the label with its gold evidence."""
    primary_schema = schema_for(primary["prompt_version"])
    primary_entries = _scored_entries(primary)
    if len(primary_entries) != len(primary["per_doc"]):
        raise ValueError(f"{REPLAY_ARTIFACT.name} holds entries that were not scored")
    later_runs = [
        (path, schema_for(a["prompt_version"]), _scored_entries(a))
        for path, a in _evidence_artifacts(primary["model"], results_dir)
    ]
    replays = {}
    for doc_id in doc_ids:
        if doc_id in primary_entries:
            replays[doc_id] = Replay(
                primary_schema, primary["model"], REPLAY_ARTIFACT, primary_entries[doc_id]
            )
            continue
        found = next(((p, s, e[doc_id]) for p, s, e in later_runs if doc_id in e), None)
        if found is None:
            replays[doc_id] = Replay(SCHEMA_V2, LABEL_REPLAY_MODEL)
        else:
            path, schema, entry = found
            replays[doc_id] = Replay(schema, primary["model"], path, entry)
    return replays


def scored_replay(
    doc_id: str, schema: ExtractionSchema, primary: dict, results_dir: Path = RESULTS_DIR
) -> Replay | None:
    """The committed answer of the primary run's model for one document under
    exactly this schema, or None when no committed run scored it so. This is
    choose_replays asked a different question: the demo seed takes whatever
    run it can for a document, while a request names its schema, and an answer
    in another schema's shape would not parse as this one's. Under the primary
    run's own schema that run answers; under a later one, the newest committed
    run of the same model under it does."""
    if primary["prompt_version"] == schema.prompt_version:
        entry = _scored_entries(primary).get(doc_id)
        return Replay(schema, primary["model"], REPLAY_ARTIFACT, entry) if entry else None
    for path, artifact in _evidence_artifacts(primary["model"], results_dir):
        if artifact["prompt_version"] == schema.prompt_version:
            entry = _scored_entries(artifact).get(doc_id)
            if entry:
                return Replay(schema, primary["model"], path, entry)
    return None
