"""Value-quote agreement (app/quotes.py): pure functions, no database."""

from decimal import Decimal

import pytest

from app.quotes import (
    STATUS_FAIL_MISMATCH,
    STATUS_FAIL_NO_NUMBER,
    STATUS_PASS,
    STATUS_SKIPPED_NO_QUOTE,
    STATUS_SKIPPED_ZERO_UNPRINTED,
    is_skipped,
    quote_agreement,
)


@pytest.mark.parametrize(
    ("value", "quote"),
    [
        ("1234.56", "Total  $1,234.56"),
        ("1234.56", "1.234,56 EUR"),
        ("5.00", "-$5.00"),
        ("-5.00", "(5.00)"),
        ("-5.00", "Discount -$5.00"),
        ("8.875", "Sales tax 8.875%"),
        ("9.00", "2 x 4.50 = 9.00"),
        ("4.50", "2 x 4.50 = 9.00"),
        ("2", "2 x 4.50 = 9.00"),
        ("46.2", "46.20"),
        ("1234", "1,234"),
        ("1.234", "1,234"),
        ("0", "Shipping 0.00"),
    ],
)
def test_a_value_the_quote_prints_agrees(value: str, quote: str) -> None:
    verdict = quote_agreement(Decimal(value), quote)

    assert verdict.status == STATUS_PASS
    assert abs(Decimal(value)) in verdict.numbers


@pytest.mark.parametrize(
    ("value", "quote", "numbers"),
    [
        ("46.25", "Subtotal 46.52", ["46.52"]),
        ("46.25", "Subtotal 46.2", ["46.2"]),
        ("46.25", "Subtotal 146.25", ["146.25"]),
        ("1234.56", "$1,234.65", ["1234.65"]),
        ("8.875", "Tax 8.5%", ["8.5"]),
        ("10.00", "2 x 4.50 = 9.00", ["2", "4.50", "9.00"]),
        ("1235", "1,234", ["1.234", "1234"]),
    ],
)
def test_a_value_the_quote_does_not_print_is_a_mismatch(
    value: str, quote: str, numbers: list[str]
) -> None:
    verdict = quote_agreement(Decimal(value), quote)

    assert verdict.status == STATUS_FAIL_MISMATCH
    assert verdict.numbers == tuple(Decimal(n) for n in numbers)


def test_the_mismatch_detail_shows_both_numbers_in_plain_words() -> None:
    one = quote_agreement(Decimal("46.25"), "Subtotal 46.52", minor_units=2)
    several = quote_agreement(Decimal(10), "2 x 4.50 = 9.00", minor_units=2)

    assert one.detail == "The value is 46.25, but its quote shows 46.52."
    assert several.detail == (
        "The value is 10.00, but none of the numbers in its quote (2.00, 4.50, 9.00) is it."
    )


def test_a_date_in_the_quote_is_not_a_number_the_detail_lists() -> None:
    verdict = quote_agreement(
        Decimal(300), "Deposit received 06/30/2026: -$250.00", minor_units=2
    )

    assert verdict.status == STATUS_FAIL_MISMATCH
    assert verdict.numbers == (Decimal("250.00"),)
    assert verdict.detail == "The value is 300.00, but its quote shows 250.00."
    assert quote_agreement(Decimal(250), "Deposit received 06/30/2026: -$250.00", minor_units=2).status == STATUS_PASS


def test_amounts_in_the_detail_follow_the_currency_places_and_quantities_stay_plain() -> None:
    assert quote_agreement(Decimal("46.2"), "46.52", minor_units=2).detail == (
        "The value is 46.20, but its quote shows 46.52."
    )
    # Digits past the places are kept, never rounded away.
    assert quote_agreement(Decimal("46.2"), "46.52", minor_units=0).detail == (
        "The value is 46.2, but its quote shows 46.52."
    )
    assert quote_agreement(Decimal("2.5"), "Qty 3").detail == "The value is 2.5, but its quote shows 3."


@pytest.mark.parametrize("quote", ["no total printed", "n/a", "$", "12.05.2024"])
def test_a_quote_with_no_number_fails(quote: str) -> None:
    verdict = quote_agreement(Decimal("5.00"), quote)

    assert verdict.status == STATUS_FAIL_NO_NUMBER
    assert verdict.numbers == ()
    assert verdict.detail == "The value is 5, but its quote contains no number."


@pytest.mark.parametrize("quote", [None, "", "   ", "\n"])
def test_no_quote_is_skipped_not_failed(quote: str | None) -> None:
    verdict = quote_agreement(Decimal("5.00"), quote)

    assert verdict.status == STATUS_SKIPPED_NO_QUOTE
    assert verdict.numbers == ()
    assert verdict.detail == "The field has no quote to check."


def test_an_ambiguous_quote_agrees_with_either_reading_and_no_other() -> None:
    assert quote_agreement(Decimal(1234), "1,234").status == STATUS_PASS
    assert quote_agreement(Decimal("1.234"), "1,234").status == STATUS_PASS
    assert quote_agreement(Decimal("12.34"), "1,234").status == STATUS_FAIL_MISMATCH


def test_the_comparison_is_exact_to_the_last_digit() -> None:
    assert quote_agreement(Decimal("0.1") + Decimal("0.2"), "0.30").status == STATUS_PASS
    assert quote_agreement(Decimal("99999999999.99"), "99,999,999,999.98").status == STATUS_FAIL_MISMATCH


@pytest.mark.parametrize(
    "quote", ["Shipping: FREE", "Tip: ____", "Tax: -", "Discount: none", "n/a"]
)
def test_a_zero_said_in_words_is_skipped_not_failed(quote: str) -> None:
    verdict = quote_agreement(Decimal("0.00"), quote)

    assert verdict.status == STATUS_SKIPPED_ZERO_UNPRINTED
    assert is_skipped(verdict)
    assert verdict.detail == "The value is 0 and its quote prints no number."


def test_only_a_zero_gets_that_allowance() -> None:
    assert quote_agreement(Decimal(5), "Shipping: FREE").status == STATUS_FAIL_NO_NUMBER
    # A zero beside a different number still disagrees.
    assert quote_agreement(Decimal(0), "Shipping: FREE (saves 5.00)").status == STATUS_FAIL_MISMATCH
    assert quote_agreement(Decimal(0), "Shipping 0.00").status == STATUS_PASS


def test_places_settle_an_ambiguous_quote() -> None:
    # A dollar amount cannot be 1.000, so "$1,000" is a thousand and 1 does not agree.
    assert quote_agreement(Decimal(1), "Total $1,000", places=2).status == STATUS_FAIL_MISMATCH
    assert quote_agreement(Decimal(1000), "Total $1,000", places=2).status == STATUS_PASS
    assert quote_agreement(Decimal("2.5"), "$2,500", places=2).status == STATUS_FAIL_MISMATCH
    # A three-decimal field keeps both readings.
    assert quote_agreement(Decimal("1.234"), "1,234", places=3).status == STATUS_PASS
    assert quote_agreement(Decimal("1.459"), "1,459 \u20ac/l", places=6).status == STATUS_PASS


@pytest.mark.parametrize(
    "quote", ["USD 1 234,56", "1\u00a0234,56 \u20ac", "1\u202f234,56 \u20ac", "CHF 1'234.56", "CHF 1\u2019234.56"]
)
def test_a_grouped_number_agrees_with_the_whole_amount(quote: str) -> None:
    assert quote_agreement(Decimal("1234.56"), quote, places=2).status == STATUS_PASS


def test_a_leading_separator_amount_agrees() -> None:
    assert quote_agreement(Decimal("0.50"), "$.50", places=2).status == STATUS_PASS
