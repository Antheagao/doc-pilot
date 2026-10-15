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
keeps its own copy of the v1 field names to stay free of this module's
imports.
"""

from dataclasses import dataclass, field
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
}


@dataclass(frozen=True)
class ExtractionSchema:
    """A prompt version and everything that has to change with it.

    `evidence` says whether the version's leaves carry the quote they were
    read from; v1 has none.

    The prompt files are read when the schema is built, at import for the
    registered ones: they are small, static files that never change at
    runtime, and a missing one must fail at startup, not after a billed call.
    """

    prompt_version: str
    prompt_path: Path
    repair_prompt_path: Path
    fields: tuple[FieldSpec, ...]
    evidence: bool
    max_tokens: int
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

SCHEMAS: dict[str, ExtractionSchema] = {SCHEMA_V1.prompt_version: SCHEMA_V1}


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


def _leaf_schema(value_schema: dict[str, Any]) -> dict[str, Any]:
    """A `{"value": <value_schema>, "confidence": number}` leaf, the shape
    every extracted field (and line-item sub-field) uses per the prompt.
    `minimum`/`maximum` on confidence are hints to the model, not an
    enforced guarantee -- tool_choice does not validate input against
    input_schema, so _coerce_leaf still has to clamp defensively.
    """
    return {
        "type": "object",
        "properties": {
            "value": value_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["value", "confidence"],
    }


_STRING_OR_NULL = {"type": ["string", "null"]}
_NUMBER_OR_NULL = {"type": ["number", "null"]}

_VALUE_SCHEMAS: dict[FieldKind, dict[str, Any]] = {
    "text": _STRING_OR_NULL,
    "date": _STRING_OR_NULL,
    "currency": _STRING_OR_NULL,
    "money": _NUMBER_OR_NULL,
    "number": _NUMBER_OR_NULL,
}


def _array_item_schema(array_name: str) -> dict[str, Any]:
    cells = ARRAY_CELLS[array_name]
    return {
        "type": "object",
        "properties": {key: _leaf_schema(_VALUE_SCHEMAS[kind]) for key, kind in cells.items()},
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
        field.name: _leaf_schema(
            {"type": "array", "items": _array_item_schema(field.name)}
            if field.kind == "array"
            else _VALUE_SCHEMAS[field.kind]
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
