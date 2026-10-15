"""Tests for app.extraction_fields: the v1 schema must stay byte-identical to
what shipped, and the registry must refuse a version it does not know.

tests/fixtures/record_extraction_tool_v1.json was captured from the module
constants this registry replaced (RECORD_EXTRACTION_TOOL, PROMPT_TEXT,
REPAIR_PROMPT_TEXT, DEFAULT_MAX_TOKENS, TOP_LEVEL_FIELDS, MONEY_FIELDS), so a
change to the request the model sees fails here instead of passing silently.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.extraction_fields import (
    ARRAY_CELLS,
    SCHEMA_V1,
    SCHEMAS,
    FieldSpec,
    money_cell_keys,
    record_extraction_tool,
    schema_for,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "record_extraction_tool_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_v1_tool_is_byte_identical_to_the_captured_tool() -> None:
    tool = record_extraction_tool(SCHEMA_V1)

    assert tool == FIXTURE["tool"]
    # Dict equality ignores key order, and the request serializes it in order.
    assert json.dumps(tool) == json.dumps(FIXTURE["tool"])


def test_v1_prompts_token_budget_and_field_sets_match_the_capture() -> None:
    assert SCHEMA_V1.prompt_version == FIXTURE["prompt_version"] == "extract_v1"
    assert SCHEMA_V1.prompt_text == FIXTURE["prompt_text"]
    assert SCHEMA_V1.repair_prompt_version == FIXTURE["repair_prompt_version"]
    assert SCHEMA_V1.repair_prompt_text == FIXTURE["repair_prompt_text"]
    assert SCHEMA_V1.max_tokens == FIXTURE["max_tokens"]
    assert list(SCHEMA_V1.field_names) == FIXTURE["top_level_fields"]
    assert list(SCHEMA_V1.money_field_names) == FIXTURE["money_fields"]
    assert list(money_cell_keys("line_items")) == FIXTURE["line_item_money_keys"]
    assert SCHEMA_V1.evidence is False


def test_v1_merge_tie_breaks_are_header_earliest_and_amounts_latest() -> None:
    tie_breaks = {field.name: field.merge_tie_break for field in SCHEMA_V1.fields}

    assert tie_breaks == {
        "vendor": "earliest",
        "document_date": "earliest",
        "line_items": None,
        "subtotal": "latest",
        "tax": "latest",
        "total": "latest",
        "currency": "earliest",
    }


def test_the_line_item_cells_are_listed_in_label_order() -> None:
    assert list(ARRAY_CELLS["line_items"]) == ["description", "quantity", "unit_price", "total"]


def test_schema_for_returns_the_registered_schema() -> None:
    assert schema_for("extract_v1") is SCHEMA_V1
    assert SCHEMAS == {"extract_v1": SCHEMA_V1}


def test_schema_for_an_unknown_version_names_the_known_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        schema_for("extract_v9")

    assert "extract_v9" in str(excinfo.value)
    assert "known versions: extract_v1" in str(excinfo.value)


def test_a_missing_prompt_file_fails_when_the_schema_is_built(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        replace(SCHEMA_V1, prompt_path=tmp_path / "extract_v9.md")
    with pytest.raises(FileNotFoundError):
        replace(SCHEMA_V1, repair_prompt_path=tmp_path / "repair_v9.md")


def test_a_field_spec_has_a_tie_break_exactly_when_it_is_not_an_array() -> None:
    with pytest.raises(ValueError, match="line_items"):
        FieldSpec("line_items", "array", "earliest")
    with pytest.raises(ValueError, match="vendor"):
        FieldSpec("vendor", "text", None)


def test_settings_default_to_the_v1_prompt_version() -> None:
    assert Settings().extraction_prompt_version == "extract_v1"


def test_settings_accept_a_registered_prompt_version() -> None:
    settings = Settings(extraction_prompt_version="extract_v1")

    assert schema_for(settings.extraction_prompt_version) is SCHEMA_V1


def test_settings_reject_an_unknown_prompt_version() -> None:
    with pytest.raises(ValidationError, match="known versions: extract_v1"):
        Settings(extraction_prompt_version="extract_v9")
