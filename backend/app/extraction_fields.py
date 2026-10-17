"""The extraction schemas: which fields a prompt version reads, and the tool
that asks the model for them.

An ExtractionSchema ties one prompt version to its field list, its repair
prompt, its token budget, and whether its leaves carry evidence. Every
reader of extracted data takes the schema of the extraction it is reading
(`schema_for(extraction.prompt_version)`), so two prompt versions can coexist
in the database and in the evals without a module-level constant deciding for
both. A schema is data only: it imports nothing from the app, so config,
evals, and scripts can all load it without a cycle.

The field kinds are the vocabulary the rest of the code keys off: money
parsing and formatting (app/money.py) follows "money", and the page merge
(app/extraction.py) follows each field's merge_tie_break. app/evals/dataset.py
takes the label's field set from SCHEMA_V2, so a label has every field the
richest schema reads.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

FieldKind = Literal["text", "date", "currency", "money", "number", "array"]
TieBreak = Literal["earliest", "latest"]

DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class FieldSpec:
    """One top-level field of the tool, in the order ExtractedField rows are
    created.

    merge_tie_break says which page wins when a chunked PDF's pages report the
    same confidence: "earliest" for header fields (a document's header is up
    front), "latest" for amounts (totals print at the end). An array field
    concatenates its pages instead, so it has none.
    """

    name: str
    kind: FieldKind
    merge_tie_break: TieBreak | None

    def __post_init__(self) -> None:
        if (self.kind == "array") != (self.merge_tie_break is None):
            raise ValueError(
                f"field {self.name!r}: an array field has no merge_tie_break "
                "and every other field has one"
            )


# The cells of each array field, in order, and their kinds. "number" is a
# count (a quantity), not money, so money parsing leaves it alone.
ARRAY_CELLS: dict[str, dict[str, FieldKind]] = {
    "line_items": {
        "description": "text",
        "quantity": "number",
        "unit_price": "money",
        "total": "money",
    },
    # A tax line's rate is the printed percent (8.875 for 8.875%), a plain
    # number that code divides by 100, so it is "number", not money.
    "tax_lines": {
        "label": "text",
        "rate": "number",
        "base": "money",
        "amount": "money",
    },
}


@dataclass(frozen=True)
class ExtractionSchema:
    """A prompt version and everything that has to change with it.

    `evidence` says whether the version's leaves carry the page and quote they
    were read from; v1 has none. The array fields a version reads must be
    listed in ARRAY_CELLS.

    The prompt files are read when the schema is built, at import for the
    registered ones: they are small, static files that never change at
    runtime, and a missing one must fail at startup, not after a billed call.

    `prompt_cache` says whether requests mark the tool and the prompt as a cache
    prefix (extraction._system_for_request). v1 does not, so its request stays
    byte-identical to the one captured before caching existed.
    """

    prompt_version: str
    prompt_path: Path
    repair_prompt_path: Path
    fields: tuple[FieldSpec, ...]
    evidence: bool
    max_tokens: int
    prompt_cache: bool = False
    prompt_text: str = field(init=False, compare=False, repr=False)
    repair_prompt_text: str = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_text", self.prompt_path.read_text(encoding="utf-8"))
        object.__setattr__(
            self, "repair_prompt_text", self.repair_prompt_path.read_text(encoding="utf-8")
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def money_field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.kind == "money")

    @property
    def repair_prompt_version(self) -> str:
        return self.repair_prompt_path.stem


def money_cell_keys(array_name: str) -> tuple[str, ...]:
    """The cells of an array field that hold an amount (a line item's unit
    price and total, not its quantity)."""
    return tuple(key for key, kind in ARRAY_CELLS[array_name].items() if kind == "money")


# The prompt version is the prompt file's stem, so new wording is a new file
# and a new version string with no separate constant to keep in step.
_V1_PROMPT_PATH = PROMPTS_DIR / "extract_v1.md"

SCHEMA_V1 = ExtractionSchema(
    prompt_version=_V1_PROMPT_PATH.stem,
    prompt_path=_V1_PROMPT_PATH,
    repair_prompt_path=PROMPTS_DIR / "repair_v1.md",
    fields=(
        FieldSpec("vendor", "text", "earliest"),
        FieldSpec("document_date", "date", "earliest"),
        FieldSpec("line_items", "array", None),
        FieldSpec("subtotal", "money", "latest"),
        FieldSpec("tax", "money", "latest"),
        FieldSpec("total", "money", "latest"),
        FieldSpec("currency", "currency", "earliest"),
    ),
    evidence=False,
    max_tokens=DEFAULT_MAX_TOKENS,
)

# v2 adds the fields real invoices carry (identifiers, terms, the payment, the
# adjustments, the balances, the tax lines) and asks every leaf for the evidence
# it was read from. Header-ish fields (ids, terms, payment, dates) break a page
# tie toward the earliest page, amounts toward the latest.
_V2_PROMPT_PATH = PROMPTS_DIR / "extract_v2.md"

SCHEMA_V2 = ExtractionSchema(
    prompt_version=_V2_PROMPT_PATH.stem,
    prompt_path=_V2_PROMPT_PATH,
    repair_prompt_path=PROMPTS_DIR / "repair_v2.md",
    fields=(
        FieldSpec("vendor", "text", "earliest"),
        FieldSpec("invoice_number", "text", "earliest"),
        FieldSpec("po_number", "text", "earliest"),
        FieldSpec("vendor_tax_id", "text", "earliest"),
        FieldSpec("document_date", "date", "earliest"),
        FieldSpec("due_date", "date", "earliest"),
        FieldSpec("payment_terms", "text", "earliest"),
        FieldSpec("payment_method", "text", "earliest"),
        FieldSpec("payment_card_last4", "text", "earliest"),
        FieldSpec("line_items", "array", None),
        FieldSpec("subtotal", "money", "latest"),
        FieldSpec("discount", "money", "latest"),
        FieldSpec("shipping", "money", "latest"),
        FieldSpec("tax", "money", "latest"),
        FieldSpec("tax_lines", "array", None),
        FieldSpec("tip", "money", "latest"),
        FieldSpec("total", "money", "latest"),
        FieldSpec("amount_paid", "money", "latest"),
        FieldSpec("amount_due", "money", "latest"),
        FieldSpec("currency", "currency", "earliest"),
    ),
    evidence=True,
    # Every leaf also carries a page and a quote, which roughly doubles what a
    # row costs in output tokens, so v2 gets four times v1's budget to keep a
    # dense invoice from stopping at max_tokens. It stays under the SDK's
    # non-streaming limit (about 21,000 tokens), so requests need no streaming.
    max_tokens=16384,
    # The tool and the prompt are about 8,000 tokens (a live call's input
    # tokens minus its image), the same on every call; the document is the
    # only part that differs, so the rest is worth caching.
    prompt_cache=True,
)

# v3 is v2 with one prompt rule added: a tax line's base is only an amount the
# line itself prints, otherwise null. Same fields, evidence, token budget, and
# caching, and the repair prompt stays repair_v2 because a repair fixes structure
# and must not change values. v2 stays registered: its stored extractions and the
# committed run artifacts name it.
_V3_PROMPT_PATH = PROMPTS_DIR / "extract_v3.md"

SCHEMA_V3 = replace(SCHEMA_V2, prompt_version=_V3_PROMPT_PATH.stem, prompt_path=_V3_PROMPT_PATH)

SCHEMAS: dict[str, ExtractionSchema] = {
    schema.prompt_version: schema for schema in (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3)
}


def schema_for(prompt_version: str) -> ExtractionSchema:
    """The schema a prompt version was extracted with. An unknown version
    raises ValueError naming the known ones, so a misspelled
    EXTRACTION_PROMPT_VERSION fails when settings load, not mid-job."""
    try:
        return SCHEMAS[prompt_version]
    except KeyError:
        raise ValueError(
            f"unknown extraction prompt version {prompt_version!r}; "
            f"known versions: {', '.join(sorted(SCHEMAS))}"
        ) from None


# The evidence of a v2 leaf: where on the document its value was read, or null
# when there is nothing to cite. `minimum` and `maxLength` are hints to the
# model, like the confidence bounds below: the tool is not a strict tool (strict
# mode drops those keywords), so extraction._coerce_evidence enforces them.
EVIDENCE_QUOTE_MAX_CHARS = 200

_EVIDENCE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
                "quote": {"type": "string", "maxLength": EVIDENCE_QUOTE_MAX_CHARS},
            },
            "required": ["page", "quote"],
        },
        {"type": "null"},
    ]
}


def _leaf_schema(value_schema: dict[str, Any], *, evidence: bool = False) -> dict[str, Any]:
    """A `{"value": <value_schema>, "confidence": number}` leaf, the shape
    every extracted field (and line-item sub-field) uses per the prompt, plus
    an `evidence` property when the schema asks for it.
    `minimum`/`maximum` on confidence are hints to the model, not an
    enforced guarantee -- tool_choice does not validate input against
    input_schema, so _coerce_leaf still has to clamp defensively.
    """
    properties: dict[str, Any] = {
        "value": value_schema,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    required = ["value", "confidence"]
    if evidence:
        properties["evidence"] = _EVIDENCE_SCHEMA
        required.append("evidence")
    return {"type": "object", "properties": properties, "required": required}


_STRING_OR_NULL = {"type": ["string", "null"]}
_NUMBER_OR_NULL = {"type": ["number", "null"]}

_VALUE_SCHEMAS: dict[FieldKind, dict[str, Any]] = {
    "text": _STRING_OR_NULL,
    "date": _STRING_OR_NULL,
    "currency": _STRING_OR_NULL,
    "money": _NUMBER_OR_NULL,
    "number": _NUMBER_OR_NULL,
}


def _array_item_schema(array_name: str, *, evidence: bool) -> dict[str, Any]:
    cells = ARRAY_CELLS[array_name]
    return {
        "type": "object",
        "properties": {
            key: _leaf_schema(_VALUE_SCHEMAS[kind], evidence=evidence)
            for key, kind in cells.items()
        },
        "required": list(cells),
    }


def record_extraction_tool(schema: ExtractionSchema) -> dict[str, Any]:
    """The extraction tool for a schema. tool_choice forces the model to call
    it (see extract_document), so a tool_use block is guaranteed -- but its
    input is not guaranteed to match this schema (see app/extraction.py's
    module docstring), so every consumer of tool_input must go through
    _coerce_leaf.
    """
    properties = {
        # An array's own leaf has no evidence: the stored array keeps only its
        # rows, and each cell cites its own quote.
        field.name: _leaf_schema(
            {"type": "array", "items": _array_item_schema(field.name, evidence=schema.evidence)}
            if field.kind == "array"
            else _VALUE_SCHEMAS[field.kind],
            evidence=schema.evidence and field.kind != "array",
        )
        for field in schema.fields
    }
    return {
        "name": "record_extraction",
        "description": (
            "Record the structured data extracted from the receipt/invoice "
            "document, with a per-field confidence score."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(schema.field_names),
        },
    }
