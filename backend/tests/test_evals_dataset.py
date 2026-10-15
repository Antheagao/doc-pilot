"""Dataset v2 (repo-root evals/): the validators for the v2 label fields and the
gold evidence, and the committed labels, text, and documents checked against the
generator that wrote them (backend/scripts/make_evals.py). No database and no
network."""

import copy
import hashlib
import importlib.util
import json
import random
import re
import sys
from decimal import Decimal

import pypdf
import pytest

from app.config import BACKEND_DIR
from app.evals.checks import failing_labels, reconcile_label
from app.evals.dataset import (
    EVALS_DIR,
    FIELD_KEYS,
    _load_case,
    _validate_card_last4,
    _validate_evidence,
    _validate_payment_method,
    _validate_tax_lines,
    load_cases,
)
from app.evals.retrieval import load_gold_corpus
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2
from app.reconcile import KIND_QUOTE, STATUS_PASS

V1_DOCUMENT_COUNT = 25
V1_DIGESTS = BACKEND_DIR / "tests" / "fixtures" / "label_v1_fields_sha256.json"


def _load_make_evals():
    spec = importlib.util.spec_from_file_location(
        "make_evals", BACKEND_DIR / "scripts" / "make_evals.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered so the script's @dataclass classes can resolve their own module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_evals = _load_make_evals()


def v1_cases():
    return [case for case in load_cases() if int(case.doc_id[:3]) <= V1_DOCUMENT_COUNT]


def synthetic_cases():
    """The generated documents (001-032). A label harvested from human review
    has no generator record and, once committed, is no part of these checks."""
    return [case for case in load_cases() if case.source == "synthetic"]


def digest(fields: dict) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---- The committed dataset ---------------------------------------------------


def test_the_v1_fields_of_the_first_25_labels_are_unchanged() -> None:
    # The digests were taken from the labels as dataset v1 committed them, before
    # any v2 field existed; the 25 images were never regenerated, so the labels
    # must still say what the images show.
    frozen = json.loads(V1_DIGESTS.read_text(encoding="utf-8"))

    current = {
        case.doc_id: digest({name: case.fields[name] for name in SCHEMA_V1.field_names})
        for case in v1_cases()
    }

    assert len(frozen) == V1_DOCUMENT_COUNT
    assert current == frozen


def test_every_label_is_dataset_v2_with_the_whole_field_set() -> None:
    cases = load_cases()

    assert len(synthetic_cases()) == 32
    assert {case.dataset_version for case in cases} == {"v2"}
    assert all(set(case.fields) == FIELD_KEYS == set(SCHEMA_V2.field_names) for case in cases)


def test_every_label_passes_the_rules_with_every_gold_quote_agreeing() -> None:
    assert failing_labels(load_cases()) == []
    for case in synthetic_cases():
        report = reconcile_label(case.fields, case.evidence)
        quote_checks = [check for check in report.checks if check.kind == KIND_QUOTE]
        assert report.status == STATUS_PASS, case.doc_id
        assert quote_checks, case.doc_id
        # Every quote judged something and agreed: none is skipped or failing.
        assert {check.status for check in quote_checks} == {STATUS_PASS}, case.doc_id


def test_every_value_of_a_synthetic_label_has_gold_evidence() -> None:
    def values(fields):
        for name in SCHEMA_V2.field_names:
            if name in ("line_items", "tax_lines"):
                for index, row in enumerate(fields[name] or []):
                    for cell, value in row.items():
                        yield f"{name}[{index}].{cell}", value
            else:
                yield name, fields[name]

    # A null value has no evidence: 031 prints tax lines and no total-tax line, so
    # its tax is null, not a sum.
    assert next(c for c in load_cases() if c.doc_id.startswith("031")).fields["tax"] is None
    for case in synthetic_cases():
        unevidenced = {path for path, value in values(case.fields) if value is not None} - set(
            case.evidence
        )
        assert unevidenced == set(), case.doc_id


def test_the_two_page_pdf_cites_page_two_for_its_totals() -> None:
    case = next(case for case in load_cases() if case.doc_id.startswith("032"))

    assert case.mime_type == "application/pdf"
    assert len(pypdf.PdfReader(case.image_path).pages) == 2
    pages = {path: entry["page"] for path, entry in case.evidence.items()}
    for path in ("vendor", "invoice_number", "document_date", "line_items[0].total", "line_items[7].total"):
        assert pages[path] == 1, path
    for path in (
        "line_items[8].total",
        "line_items[11].total",
        "subtotal",
        "shipping",
        "tax",
        "tax_lines[0].rate",
        "total",
        "amount_due",
        "currency",
    ):
        assert pages[path] == 2, path


def test_the_card_number_receipt_keeps_only_the_brand_and_last_four_digits() -> None:
    case = next(case for case in load_cases() if case.doc_id.startswith("029"))
    label_text = (EVALS_DIR / "labels" / f"{case.doc_id}.json").read_text(encoding="utf-8")

    assert (case.fields["payment_method"], case.fields["payment_card_last4"]) == ("VISA", "1111")
    assert case.evidence["payment_method"]["quote"] == "Paid by VISA ****1111 $88.57"
    assert case.evidence["amount_paid"]["quote"] == case.evidence["payment_card_last4"]["quote"]
    assert "4111" not in label_text


def test_no_label_holds_a_run_of_thirteen_digits() -> None:
    # Spaces and dashes may separate the digits of a card number, so count those too.
    run = re.compile(r"\d(?:[ \-]?\d){12}")
    for path in (EVALS_DIR / "labels").glob("*.json"):
        assert run.search(path.read_text(encoding="utf-8")) is None, path.name


def test_the_gold_corpus_is_still_the_25_documents_of_dataset_v1() -> None:
    corpus = load_gold_corpus(load_cases())

    assert [doc.doc_id for doc in corpus] == [case.doc_id for case in v1_cases()]
    assert len(corpus) == V1_DOCUMENT_COUNT


# ---- The generator and the committed files ------------------------------------


def test_the_generator_reproduces_every_committed_label() -> None:
    committed = {case.doc_id: case for case in synthetic_cases()}

    assert {spec.stem for spec in make_evals.DOC_SPECS} == set(committed)
    for spec in make_evals.DOC_SPECS:
        _, _, label = make_evals._record_and_label(spec)
        path = EVALS_DIR / "labels" / f"{spec.stem}.json"
        assert json.loads(json.dumps(label)) == json.loads(path.read_text(encoding="utf-8")), spec.stem
        assert path.read_text(encoding="utf-8") == json.dumps(label, indent=2) + "\n", spec.stem


def test_the_generator_reproduces_the_gold_text_of_the_25_exactly() -> None:
    # The printed strings of 001-025 are what their committed images show.
    for spec in make_evals.DOC_SPECS:
        text_path = EVALS_DIR / "text" / f"{spec.stem}.txt"
        assert text_path.is_file() is spec.gold_text, spec.stem
        if spec.gold_text:
            record = make_evals.build_record(spec, random.Random(spec.index))
            assert make_evals.render_text(spec, record) == text_path.read_text(encoding="utf-8")


def test_text_only_writes_text_for_the_25_and_none_for_the_new_documents(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(make_evals, "EVALS_DIR", tmp_path / "evals")
    monkeypatch.setattr(make_evals, "TEXT_DIR", tmp_path / "evals" / "text")

    assert make_evals.write_text_only(make_evals.DOC_SPECS) == 0

    written = sorted(path.name for path in (tmp_path / "evals" / "text").iterdir())
    assert written == sorted(f"{spec.stem}.txt" for spec in make_evals.DOC_SPECS if spec.gold_text)
    assert len(written) == V1_DOCUMENT_COUNT


def test_every_gold_quote_is_a_printed_line_as_the_product_would_store_it() -> None:
    for spec in make_evals.DOC_SPECS:
        record = make_evals.build_record(spec, random.Random(spec.index))
        printed = make_evals.build_text_lines(spec, record)
        evidence = make_evals.build_evidence(record, printed)
        assert evidence, spec.stem
        for path, entry in evidence.items():
            index = printed.line_of[path]
            assert entry["page"] == printed.pages[index], (spec.stem, path)
            # The printed line, stripped, with a card number masked and nothing else changed.
            line = printed.lines[index][0]
            assert entry["quote"] == make_evals._gold_quote(path, line), (spec.stem, path)
            if path not in make_evals._PAYMENT_PATHS and "amount_paid" != path:
                assert entry["quote"] in line, (spec.stem, path)


def test_the_new_documents_are_computed_in_decimal_rounded_half_up() -> None:
    assert make_evals._cents(Decimal("2.675")) == Decimal("2.68")
    assert make_evals._cents(Decimal("0.125")) == Decimal("0.13")
    fields = {case.doc_id[:3]: case.fields for case in load_cases()}

    # 026: 478.40 less 5% (23.92) plus 18.50 freight and two taxes on the
    # discounted 454.48: 6.5% is 29.5412 and 1.25% is 5.681.
    assert fields["026"]["discount"] == 23.92
    assert [row["amount"] for row in fields["026"]["tax_lines"]] == [29.54, 5.68]
    assert (fields["026"]["tax"], fields["026"]["total"]) == (35.22, 508.2)
    # 027: 8.25% of 71.00 is 5.8575, and the tip is part of the total.
    assert (fields["027"]["tax"], fields["027"]["tip"], fields["027"]["total"]) == (5.86, 14.0, 90.86)
    # 028: 564.98 less a 250.00 deposit.
    assert (fields["028"]["amount_paid"], fields["028"]["amount_due"]) == (250.0, 314.98)
    # 030: each VAT group is discounted before its own VAT.
    assert [row["base"] for row in fields["030"]["tax_lines"]] == [125.19, 118.8]
    assert [row["amount"] for row in fields["030"]["tax_lines"]] == [23.79, 8.32]
    # 031: three tax lines and no total-tax line, so the tax is null and the total
    # still includes the lines (1,018.00 + 40.72 + 25.45 + 10.18).
    assert fields["031"]["tax"] is None and len(fields["031"]["tax_lines"]) == 3
    assert fields["031"]["total"] == 1094.35
    # 032: 7.25% of 1,997.00 is 144.7825.
    assert fields["032"]["tax_lines"][0]["amount"] == 144.78


def test_the_two_page_pdf_renders_byte_identically(tmp_path) -> None:
    spec = next(spec for spec in make_evals.DOC_SPECS if spec.ext == "pdf")
    paths = []
    for run in ("first", "second"):
        # The same file name in two directories: Pillow puts the name in the PDF.
        (tmp_path / run).mkdir()
        paths.append(tmp_path / run / "invoice.pdf")
        rng, record, _ = make_evals._record_and_label(spec)
        make_evals._save_images(make_evals.render_document(spec, record, rng), paths[-1], "pdf")

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert len(pypdf.PdfReader(paths[0]).pages) == 2


# ---- --labels-only -------------------------------------------------------------


@pytest.fixture
def scratch_dataset(tmp_path, monkeypatch):
    """The generator pointed at an empty scratch evals directory."""
    for name, path in (
        ("EVALS_DIR", tmp_path),
        ("DOCS_DIR", tmp_path / "docs"),
        ("LABELS_DIR", tmp_path / "labels"),
    ):
        monkeypatch.setattr(make_evals, name, path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "labels").mkdir()
    return tmp_path


def v1_label_file(spec, **overrides) -> dict:
    """The label of a dataset-v1 doc as dataset v1 wrote it: seven fields, no
    evidence."""
    _, record, label = make_evals._record_and_label(spec)
    return {
        **{k: v for k, v in label.items() if k != "evidence"},
        "dataset_version": "v1",
        "fields": {**{name: record[name] for name in SCHEMA_V1.field_names}, **overrides},
    }


def test_labels_only_rewrites_a_v1_label_as_v2_without_moving_a_v1_field(scratch_dataset) -> None:
    spec = make_evals.DOC_SPECS[0]
    (scratch_dataset / "docs" / f"{spec.stem}.{spec.ext}").write_bytes(b"not rewritten")
    label_path = scratch_dataset / "labels" / f"{spec.stem}.json"
    label_path.write_text(json.dumps(v1_label_file(spec)), encoding="utf-8")

    assert make_evals.write_labels_only((spec,)) == 0

    rewritten = json.loads(label_path.read_text(encoding="utf-8"))
    assert rewritten["dataset_version"] == "v2"
    assert set(rewritten["fields"]) == FIELD_KEYS
    assert rewritten["evidence"]
    assert (scratch_dataset / "docs" / f"{spec.stem}.{spec.ext}").read_bytes() == b"not rewritten"


def test_labels_only_refuses_when_a_v1_value_would_change(scratch_dataset, capsys) -> None:
    spec = make_evals.DOC_SPECS[0]
    (scratch_dataset / "docs" / f"{spec.stem}.{spec.ext}").write_bytes(b"image")
    label_path = scratch_dataset / "labels" / f"{spec.stem}.json"
    committed = json.dumps(v1_label_file(spec, vendor="Somebody Else"))
    label_path.write_text(committed, encoding="utf-8")

    assert make_evals.write_labels_only((spec,)) == 1

    assert "refusing to move a value the image shows" in capsys.readouterr().err
    assert label_path.read_text(encoding="utf-8") == committed


def test_labels_only_skips_a_doc_whose_image_does_not_exist_yet(scratch_dataset, capsys) -> None:
    spec = next(spec for spec in make_evals.DOC_SPECS if spec.scenario is not None)

    assert make_evals.write_labels_only((spec,)) == 0

    assert "no image yet" in capsys.readouterr().out
    assert list((scratch_dataset / "labels").iterdir()) == []


def scenario_spec():
    return next(spec for spec in make_evals.DOC_SPECS if spec.stem.startswith("026"))


def committed_scenario_label(scratch_dataset, **field_overrides) -> tuple:
    """026's committed label copied into the scratch dataset (with an image
    stand-in), `field_overrides` applied to its fields; returns (spec, path, the
    text written)."""
    spec = scenario_spec()
    label = json.loads((EVALS_DIR / "labels" / f"{spec.stem}.json").read_text(encoding="utf-8"))
    label["fields"].update(field_overrides)
    (scratch_dataset / "docs" / f"{spec.stem}.{spec.ext}").write_bytes(b"image")
    path = scratch_dataset / "labels" / f"{spec.stem}.json"
    text = json.dumps(label, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return spec, path, text


def test_labels_only_refuses_when_a_scenario_label_would_stop_matching_its_image(
    scratch_dataset, capsys
) -> None:
    # Say the shipping the generator now writes were edited after the image was drawn.
    spec, path, committed = committed_scenario_label(scratch_dataset, shipping=20.0)

    assert make_evals.write_labels_only((spec,)) == 1

    err = capsys.readouterr().err
    assert "refusing to move a value the image shows" in err and "shipping" in err
    assert path.read_text(encoding="utf-8") == committed


def test_labels_only_refuses_when_only_a_scenario_label_evidence_differs(scratch_dataset, capsys) -> None:
    spec, path, committed = committed_scenario_label(scratch_dataset)
    label = json.loads(committed)
    label["evidence"]["shipping"]["quote"] = "Shipping $18.50"
    path.write_text(json.dumps(label), encoding="utf-8")

    assert make_evals.write_labels_only((spec,)) == 1

    assert "['evidence']" in capsys.readouterr().err


def test_labels_only_accepts_a_scenario_change_only_when_told_to(scratch_dataset) -> None:
    spec, path, _ = committed_scenario_label(scratch_dataset, shipping=20.0)

    assert make_evals.write_labels_only((spec,), accept_scenario_change=True) == 0

    assert json.loads(path.read_text(encoding="utf-8"))["fields"]["shipping"] == 18.5


def test_labels_only_leaves_an_unchanged_scenario_label_as_it_is(scratch_dataset) -> None:
    spec, path, committed = committed_scenario_label(scratch_dataset)

    assert make_evals.write_labels_only((spec,)) == 0

    assert path.read_text(encoding="utf-8") == committed


def test_a_label_that_fails_the_v2_rules_is_not_written_off_as_valid(scratch_dataset) -> None:
    spec = next(spec for spec in make_evals.DOC_SPECS if spec.stem.startswith("026"))
    _, _, label = make_evals._record_and_label(spec)
    label["fields"]["discount"] = 0.0  # the total no longer adds up
    (scratch_dataset / "docs" / f"{spec.stem}.{spec.ext}").write_bytes(b"image")
    path = scratch_dataset / "labels" / f"{spec.stem}.json"
    path.write_text(json.dumps(label), encoding="utf-8")

    errors = make_evals.validate_label(path)

    assert len(errors) == 1
    assert "total_math" in errors[0]


# ---- validators ----------------------------------------------------------------


def a_label_fields() -> dict:
    return copy.deepcopy(next(case for case in load_cases() if case.doc_id.startswith("026")).fields)


@pytest.mark.parametrize("value", [None, "4242", "0000"])
def test_last4_accepts_null_and_exactly_four_digits(value) -> None:
    _validate_card_last4("doc", value)


@pytest.mark.parametrize("value", ["424", "42424", "4111 1111 1111 1111", "abcd", 4242, "４２４２", ""])
def test_last4_rejects_anything_else(value) -> None:
    with pytest.raises(ValueError, match="payment_card_last4"):
        _validate_card_last4("doc", value)


def test_payment_method_is_a_brand_not_a_number() -> None:
    _validate_payment_method("doc", "VISA")
    _validate_payment_method("doc", None)
    with pytest.raises(ValueError, match="card or account number"):
        _validate_payment_method("doc", "VISA 4111 1111 1111 1111")
    with pytest.raises(ValueError, match="string"):
        _validate_payment_method("doc", 4)


def test_tax_lines_have_exactly_label_rate_base_and_amount() -> None:
    row = {"label": "Tax", "rate": 8.0, "base": 10.0, "amount": 0.8}
    _validate_tax_lines("doc", None)
    _validate_tax_lines("doc", [])
    _validate_tax_lines("doc", [row, {**row, "label": None, "rate": None, "base": None}])
    with pytest.raises(ValueError, match="exactly keys"):
        _validate_tax_lines("doc", [{"label": "Tax", "amount": 0.8}])
    with pytest.raises(ValueError, match="exactly keys"):
        _validate_tax_lines("doc", [{**row, "extra": 1}])
    with pytest.raises(ValueError, match="rate must be numeric"):
        _validate_tax_lines("doc", [{**row, "rate": "8"}])
    with pytest.raises(ValueError, match="amount must be numeric"):
        _validate_tax_lines("doc", [{**row, "amount": True}])
    with pytest.raises(ValueError, match="label must be a string"):
        _validate_tax_lines("doc", [{**row, "label": 3}])
    with pytest.raises(ValueError, match="list or null"):
        _validate_tax_lines("doc", {"label": "Tax"})
    with pytest.raises(ValueError, match="must be an object"):
        _validate_tax_lines("doc", ["Tax"])


def test_evidence_is_optional_and_names_known_paths_with_a_page_and_a_quote() -> None:
    fields = a_label_fields()
    good = {"page": 1, "quote": "Subtotal: $478.40"}

    assert _validate_evidence("doc", None, fields) == {}
    assert _validate_evidence("doc", {}, fields) == {}
    assert _validate_evidence("doc", {"subtotal": good, "line_items[3].total": good}, fields)
    assert _validate_evidence("doc", {"tax_lines[1].rate": good}, fields)
    for path in ("nope", "line_items[9].total", "line_items[0].nope", "line_items.total", "tax_lines[2].amount"):
        with pytest.raises(ValueError, match="unknown path"):
            _validate_evidence("doc", {path: good}, fields)
    with pytest.raises(ValueError, match="value is null"):
        _validate_evidence("doc", {"tip": good}, fields)
    with pytest.raises(ValueError, match="must be an object"):
        _validate_evidence("doc", [good], fields)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"page": 0, "quote": "x"}, "page must be an integer from 1"),
        ({"page": -1, "quote": "x"}, "page must be an integer from 1"),
        ({"page": True, "quote": "x"}, "page must be an integer from 1"),
        ({"page": "1", "quote": "x"}, "page must be an integer from 1"),
        ({"page": None, "quote": "x"}, "page must be an integer from 1"),
        ({"page": 1, "quote": ""}, "non-empty"),
        ({"page": 1, "quote": "   "}, "non-empty"),
        ({"page": 1, "quote": 4}, "non-empty"),
        ({"page": 1, "quote": "x" * 201}, "over 200 characters"),
        ({"page": 1, "quote": "VISA 4111 1111 1111 1111"}, "holds a card number"),
        ({"page": 1}, "object with page and quote"),
        ({"page": 1, "quote": "x", "extra": 1}, "object with page and quote"),
        ("Subtotal", "object with page and quote"),
    ],
)
def test_a_malformed_evidence_entry_is_rejected_loudly(entry, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_evidence("doc", {"subtotal": entry}, a_label_fields())


def test_a_quote_page_must_be_a_page_the_document_has() -> None:
    fields = a_label_fields()
    on = lambda page: {"subtotal": {"page": page, "quote": "Subtotal $478.40"}}

    assert _validate_evidence("doc", on(1), fields)
    with pytest.raises(ValueError, match="past the document's 1 page"):
        _validate_evidence("doc", on(2), fields)
    assert _validate_evidence("doc", on(2), fields, page_count=2)
    with pytest.raises(ValueError, match="past the document's 2 page"):
        _validate_evidence("doc", on(3), fields, page_count=2)


def test_the_committed_pdf_label_is_checked_against_its_real_page_count(tmp_path) -> None:
    case = next(case for case in load_cases() if case.doc_id.startswith("032"))
    label = json.loads((EVALS_DIR / "labels" / f"{case.doc_id}.json").read_text(encoding="utf-8"))
    label["evidence"]["total"]["page"] = 3
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / case.image_path.name).write_bytes(case.image_path.read_bytes())
    path = tmp_path / f"{case.doc_id}.json"
    path.write_text(json.dumps(label), encoding="utf-8")

    with pytest.raises(ValueError, match="past the document's 2 page"):
        _load_case(path, evals_dir=tmp_path)


def test_a_payment_quote_gets_the_stricter_mask_the_product_applies() -> None:
    fields = a_label_fields()
    fields["payment_method"] = "VISA"
    # Five digits in a row is a card or account number to the payment fields' mask
    # even though it is far from card length and passes no Luhn check.
    quote = {"page": 1, "quote": "Paid by VISA 41111"}

    with pytest.raises(ValueError, match="holds a card number"):
        _validate_evidence("doc", {"payment_method": quote}, fields)
    with pytest.raises(ValueError, match="holds a card number"):
        _validate_evidence("doc", {"payment_card_last4": quote}, {**fields, "payment_card_last4": "1111"})
    assert _validate_evidence("doc", {"subtotal": quote}, fields)
    assert _validate_evidence(
        "doc", {"payment_method": {"page": 1, "quote": "Paid by VISA ****1111 $88.57"}}, fields
    )


def test_an_order_reference_that_is_not_a_card_number_may_be_quoted() -> None:
    quote = "Order ref 1234567890123456"

    assert _validate_evidence("doc", {"subtotal": {"page": 1, "quote": quote}}, a_label_fields())


def test_a_whole_label_with_malformed_evidence_fails_to_load(tmp_path, monkeypatch) -> None:
    case = load_cases()[0]
    label = json.loads((EVALS_DIR / "labels" / f"{case.doc_id}.json").read_text(encoding="utf-8"))
    label["evidence"]["subtotal"]["page"] = 0
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / f"{case.doc_id}.png").write_bytes(b"x")
    path = tmp_path / f"{case.doc_id}.json"
    path.write_text(json.dumps(label), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence"):
        _load_case(path, evals_dir=tmp_path)
