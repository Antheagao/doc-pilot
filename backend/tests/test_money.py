"""Money parsing and formatting (app/money.py): pure functions, no database."""

from decimal import Decimal

import pytest

from app.money import (
    ISO_4217_MINOR_UNITS,
    MoneyError,
    currency_precision,
    fits_minor_unit,
    parse_amount_input,
    parse_quantity_input,
    to_api_string,
    to_decimal,
    to_label_number,
    to_plain_string,
)


def test_iso_table_has_the_current_codes_and_their_digits() -> None:
    assert len(ISO_4217_MINOR_UNITS) > 150
    assert ISO_4217_MINOR_UNITS["USD"] == 2
    assert ISO_4217_MINOR_UNITS["JPY"] == 0
    assert {ISO_4217_MINOR_UNITS[code] for code in ("BHD", "KWD", "IQD")} == {3}
    assert ISO_4217_MINOR_UNITS["CLF"] == 4
    # Codes with no minor unit (metals, XXX) are not in the table.
    assert "XAU" not in ISO_4217_MINOR_UNITS
    assert "XXX" not in ISO_4217_MINOR_UNITS


def test_currency_precision_normalizes_and_states_its_assumptions() -> None:
    known = currency_precision("  jpy ")
    assert (known.code, known.minor_units, known.assumption) == ("JPY", 0, None)

    for missing in (None, "", "   "):
        precision = currency_precision(missing)
        assert precision.code is None and precision.minor_units == 2
        assert precision.assumption is not None and "no currency" in precision.assumption

    unknown = currency_precision("zzz")
    assert unknown.code == "ZZZ" and unknown.minor_units == 2
    assert unknown.assumption is not None and "ZZZ" in unknown.assumption


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, Decimal(5)),
        (0.1, Decimal("0.1")),
        (10.8, Decimal("10.8")),
        ("45.25", Decimal("45.25")),
        (" -3 ", Decimal(-3)),
        ("0.5", Decimal("0.5")),
        (Decimal("2.50"), Decimal("2.50")),
    ],
)
def test_to_decimal_accepts_plain_numbers(value, expected) -> None:
    assert to_decimal(value) == expected


def test_to_decimal_uses_the_shortest_float_repr_not_the_binary_value() -> None:
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal(0.1) + to_decimal(0.2) == Decimal("0.3")


@pytest.mark.parametrize(
    "value",
    [
        "1e3",
        "1E3",
        "1,234.50",
        "",
        "   ",
        "$5",
        "5 0",
        "٥",  # Arabic-Indic digit five
        "१.३",  # Devanagari 1.3
        "４５",  # fullwidth 45
        "+5",
        ".5",
        "5.",
        "--5",
        "NaN",
        "Infinity",
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        [1],
        {"value": 1},
    ],
)
def test_to_decimal_rejects_everything_else(value) -> None:
    with pytest.raises(MoneyError):
        to_decimal(value)


def test_money_error_is_a_value_error() -> None:
    assert issubclass(MoneyError, ValueError)


@pytest.mark.parametrize(
    ("value", "minor_units", "expected"),
    [
        ("46.250", 2, True),
        ("46.255", 2, False),
        ("46", 0, True),
        ("46.5", 0, False),
        ("0.0005", 3, False),
        ("0.500", 3, True),
        ("1E+3", 2, True),
        ("-12.3456", 4, True),
        ("-12.34567", 4, False),
    ],
)
def test_fits_minor_unit(value, minor_units, expected) -> None:
    assert fits_minor_unit(Decimal(value), minor_units) is expected


def test_fits_minor_unit_does_not_round_past_context_precision() -> None:
    long_tail = Decimal("1." + "0" * 40 + "1")
    assert fits_minor_unit(long_tail, 2) is False


@pytest.mark.parametrize(
    ("value", "minor_units", "expected"),
    [
        ("46.2", 2, "46.20"),
        ("46.255", 2, "46.255"),
        ("46.250", 2, "46.25"),
        ("-0", 2, "0.00"),
        ("-0.00", 2, "0.00"),
        ("0", 3, "0.000"),
        ("1500", 0, "1500"),
        ("1E+3", 2, "1000.00"),
        ("1E-7", 2, "0.0000001"),
        ("-3.5", 3, "-3.500"),
        ("12345678901234567890.5", 2, "12345678901234567890.50"),
    ],
)
def test_to_api_string(value, minor_units, expected) -> None:
    assert to_api_string(Decimal(value), minor_units) == expected


def test_to_api_string_examples_from_the_spec() -> None:
    assert to_api_string(Decimal("46.2"), 2) == "46.20"
    assert to_api_string(Decimal("46.255"), 2) == "46.255"
    assert to_api_string(Decimal("-0"), 2) == "0.00"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", "2"),
        ("2.0", "2"),
        ("2.50", "2.5"),
        ("0.125", "0.125"),
        ("-0", "0"),
        ("1E+3", "1000"),
        ("-1.5", "-1.5"),
    ],
)
def test_quantities_use_to_plain_string(value, expected) -> None:
    assert to_plain_string(Decimal(value)) == expected


def test_parse_amount_input_accepts_strings_and_numbers_that_fit() -> None:
    assert parse_amount_input("45.25", 2) == Decimal("45.25")
    assert parse_amount_input(" 45 ", 2) == Decimal(45)
    assert parse_amount_input(45.25, 2) == Decimal("45.25")
    assert parse_amount_input("1500", 0) == Decimal(1500)
    assert parse_amount_input("-0.125", 3) == Decimal("-0.125")


def test_parse_amount_input_explains_too_many_decimals() -> None:
    with pytest.raises(MoneyError, match=r"45\.255.*at most 2 decimal places"):
        parse_amount_input("45.255", 2)
    with pytest.raises(MoneyError, match="no decimal places"):
        parse_amount_input("1500.5", 0)
    with pytest.raises(MoneyError, match=r"at most 1 decimal place\."):
        parse_amount_input("1.25", 1)


def test_parse_amount_input_names_the_currency_when_given() -> None:
    jpy = currency_precision("jpy")
    with pytest.raises(MoneyError, match=r"JPY allows no decimal places\."):
        parse_amount_input("1500.5", jpy.minor_units, jpy)
    unstated = currency_precision(None)
    with pytest.raises(MoneyError, match="no currency is stated, so amounts allow at most 2"):
        parse_amount_input("1.255", unstated.minor_units, unstated)
    unlisted = currency_precision("zzz")
    with pytest.raises(MoneyError, match="'ZZZ' is not an ISO 4217 currency, so amounts allow"):
        parse_amount_input("1.255", unlisted.minor_units, unlisted)


def test_error_messages_cut_a_long_echoed_value() -> None:
    with pytest.raises(MoneyError) as caught:
        to_decimal("x" * 500)
    assert len(str(caught.value)) < 250
    assert "..." in str(caught.value)


def test_parse_quantity_input_accepts_fractions_and_returns() -> None:
    assert parse_quantity_input("2") == Decimal(2)
    assert parse_quantity_input(" 2.5 ") == Decimal("2.5")
    assert parse_quantity_input(0.125) == Decimal("0.125")
    assert parse_quantity_input("-3") == Decimal(-3)
    assert parse_quantity_input("0.000001") == Decimal("0.000001")
    assert parse_quantity_input("999999999999.5") == Decimal("999999999999.5")


def test_parse_quantity_input_bounds_precision_and_magnitude() -> None:
    with pytest.raises(MoneyError, match="quantities allow at most 6"):
        parse_quantity_input("0.0000001")
    with pytest.raises(MoneyError, match="quantities must be below"):
        parse_quantity_input("1000000000000")
    with pytest.raises(MoneyError, match="quantities must be below"):
        parse_quantity_input("-1" + "0" * 12)
    with pytest.raises(MoneyError, match="quantities must be below"):
        parse_quantity_input("9" * 50)


@pytest.mark.parametrize("raw", ["1e308", "2,5", "2 5", "", "abc", True, None, float("inf")])
def test_parse_quantity_input_rejects_what_to_decimal_rejects(raw) -> None:
    with pytest.raises(MoneyError):
        parse_quantity_input(raw)


def test_parse_amount_input_bounds_the_magnitude() -> None:
    assert parse_amount_input("999999999999.99", 2) == Decimal("999999999999.99")
    with pytest.raises(MoneyError, match="too large"):
        parse_amount_input("1000000000000", 2)
    with pytest.raises(MoneyError, match="too large"):
        parse_amount_input("-1000000000000", 2)


@pytest.mark.parametrize("raw", ["1e3", "1,234.50", "", "abc", True, None])
def test_parse_amount_input_rejects_what_to_decimal_rejects(raw) -> None:
    with pytest.raises(MoneyError):
        parse_amount_input(raw, 2)


def test_float_noise_is_not_a_valid_input() -> None:
    with pytest.raises(MoneyError, match="too many decimal places"):
        parse_amount_input(0.1 + 0.2, 2)


def test_to_label_number_is_int_when_whole_else_float() -> None:
    whole = to_label_number(Decimal("10.00"))
    assert whole == 10 and isinstance(whole, int)
    fractional = to_label_number(Decimal("10.8"))
    assert fractional == 10.8 and isinstance(fractional, float)
    zero = to_label_number(Decimal("-0.0"))
    assert zero == 0 and isinstance(zero, int)


@pytest.mark.parametrize("text", ["0.1", "10.8", "425.58", "999999999999.99", "0.005", "-46.25"])
def test_label_numbers_round_trip(text) -> None:
    assert to_decimal(to_label_number(Decimal(text))) == Decimal(text)
