"""The digit-run rules of app/pii.py: pure functions, no database."""

import time
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.pii import (
    digits_of,
    has_long_digit_run,
    mask_card_numbers,
    mask_digit_runs,
    passes_luhn,
)


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        # A card number in any grouping.
        ("VISA 4111 1111 1111 1111", "VISA ****1111"),
        ("4111-1111-1111-1111", "****1111"),
        ("4111.1111.1111.1111", "****1111"),
        ("4111111111111111", "****1111"),
        ("Amex 3782 822463 10005", "Amex ****0005"),
        ("Maestro 6759 6498 2643 8453 237", "Maestro ****3237"),
        ("Card: 4111 1111 1111 1111 exp 12/29", "Card: ****1111 exp 12/29"),
        # Fullwidth digits are digits.
        ("４１１１ １１１１ １１１１ １１１１", "****1111"),
        # Five digits in a row are a number, not a date or an amount.
        ("Auth 123456", "Auth ****3456"),
        ("acct 12345", "acct ****2345"),
        # Dates, amounts, quantities, and phone numbers are left alone.
        ("Paid 2026-10-15 VISA 1234", "Paid 2026-10-15 VISA 1234"),
        ("Paid 10/15/2026 by VISA ****1111", "Paid 10/15/2026 by VISA ****1111"),
        ("Total $1,234.56", "Total $1,234.56"),
        ("Total 1.234,56 EUR", "Total 1.234,56 EUR"),
        ("Tel 555-123-4567", "Tel 555-123-4567"),
        ("Tel (555) 123-4567", "Tel (555) 123-4567"),
        ("ending 4242", "ending 4242"),
        ("XXXX XXXX XXXX 4242", "XXXX XXXX XXXX 4242"),
        ("", ""),
    ],
)
def test_mask_digit_runs(text: str, masked: str) -> None:
    assert mask_digit_runs(text) == masked
    assert has_long_digit_run(text) is (masked != text)


def test_passes_luhn() -> None:
    assert passes_luhn("4111111111111111")
    assert passes_luhn("378282246310005")
    assert not passes_luhn("4111111111111112")
    assert not passes_luhn("1234567890123456")


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("VISA 4111 1111 1111 1111 APPROVED $50.00", "VISA ****1111 APPROVED $50.00"),
        ("Amex 3782-822463-10005", "Amex ****0005"),
        # 16 digits that fail the check are an order reference, not a card.
        ("Order 1234 5678 9012 3456 total $50.00", "Order 1234 5678 9012 3456 total $50.00"),
        # Too short or too long to be a card.
        ("Auth 123456", "Auth 123456"),
        ("1234 5678 9012 3456 7890 1234", "1234 5678 9012 3456 7890 1234"),
        ("Paid 2026-10-15 $1,234.56", "Paid 2026-10-15 $1,234.56"),
        # A card beside another group on its line is still found, and the group stays.
        ("VISA 4111 1111 1111 1111 12-27", "VISA ****1111 12-27"),
        ("VISA 4111 1111 1111 1111 12/27", "VISA ****1111 12/27"),
        ("VISA 4111 1111 1111 1111  Exp 12/27", "VISA ****1111  Exp 12/27"),
        ("4111111111111111 01/28", "****1111 01/28"),
        ("4111111111111111 0128", "****1111 0128"),
        ("4111 1111 1111 1111\t12/27", "****1111\t12/27"),
        ("4111 1111 1111 1111 123456", "****1111 123456"),
        ("Auth 123456 4111 1111 1111 1111", "Auth 123456 ****1111"),
        ("Paid 4111 1111 1111 1111 45.00", "Paid ****1111 45.00"),
        ("4111 1111 1111 1111 2026-10-15", "****1111 2026-10-15"),
        ("2026-10-15 4111 1111 1111 1111", "2026-10-15 ****1111"),
        ("AMEX 3782 822463 10005 12/27", "AMEX ****0005 12/27"),
        ("AMEX 3782 822463 10005 123456", "AMEX ****0005 123456"),
        ("4111 1111 1111 1111 1", "****1111 1"),
        ("Card 4111-1111-1111-1111 Ref 20261015", "Card ****1111 Ref 20261015"),
        # More than one other group beside it, or a longer one: the card is still found.
        ("4111 1111 1111 1111 1227 123456", "****1111 1227 123456"),
        ("4111 1111 1111 1111 12-27 123456", "****1111 12-27 123456"),
        ("4111111111111111 1227 123456 45.00", "****1111 1227 123456 45.00"),
        ("3782 822463 10005 12-27 123456", "****0005 12-27 123456"),
        ("4111 1111 1111 1111 1234567", "****1111 1234567"),
        ("4111 1111 1111 1111 1234567890", "****1111 1234567890"),
        ("1234567890 4111 1111 1111 1111", "1234567890 ****1111"),
        # A Visa test number followed by eight more digits: masked, the safe direction.
        ("4111 1111 1111 1111 1111 1111", "****1111 1111 1111"),
        # Two cards on a line are both found.
        ("4111 1111 1111 1111 5500 0000 0000 0004", "****1111 ****0004"),
        ("4111111111111111 5500000000000004", "****1111 ****0004"),
        # A card grouped like one wins over a chance match that takes in its neighbor.
        ("VISA 4821 9935 1819 0934 123456", "VISA ****0934 123456"),
        # Thin space and narrow no-break space group a card like a space does.
        ("4111\u20091111\u20091111\u20091111", "****1111"),
        ("4111\u202f1111\u202f1111\u202f1111 12/27", "****1111 12/27"),
        # Two dates joined by a dash are not a sixteen-digit card, however the check falls.
        ("Period 2026-10-01 - 2026-10-31", "Period 2026-10-01 - 2026-10-31"),
        ("Service period 01.10.2026 - 31.10.2026", "Service period 01.10.2026 - 31.10.2026"),
        ("2026-10-15 14.32.07", "2026-10-15 14.32.07"),
        ("Invoice 2026-10-15 Due 2026-11-14", "Invoice 2026-10-15 Due 2026-11-14"),
        ("Tel 555-123-4567 Fax 555-765-4321", "Tel 555-123-4567 Fax 555-765-4321"),
        ("Order ref 1234567890123456", "Order ref 1234567890123456"),
    ],
)
def test_mask_card_numbers(text: str, masked: str) -> None:
    assert mask_card_numbers(text) == masked


def _card(prefix: list[int], separator: str) -> str:
    """A card number built from `prefix` digits plus the check digit that makes
    it pass the Luhn check, printed in groups of four."""
    body = "".join(map(str, prefix))
    digits = next(body + str(check) for check in range(10) if passes_luhn(body + str(check)))
    return separator.join(digits[i : i + 4] for i in range(0, len(digits), 4)) if separator else digits


_cards = st.builds(
    _card,
    st.lists(st.integers(0, 9), min_size=15, max_size=15).map(lambda ds: [ds[0] or 4, *ds[1:]]),
    st.sampled_from(["", " ", "-"]),
)
_dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31)).flatmap(
    lambda day: st.sampled_from(["%Y-%m-%d", "%Y.%m.%d", "%d-%m-%Y", "%d.%m.%Y"]).map(day.strftime)
)
_amounts = st.from_regex(r"[0-9]{1,3}\.[0-9]{2}", fullmatch=True)
# Not three digits: a card grouped 4-4-4-4 followed by three digits is also the
# 4-4-4-4-3 grouping of a nineteen-digit card, which can win on a chance check.
_numbers = st.from_regex(r"[0-9]{1,10}", fullmatch=True).filter(lambda number: len(number) != 3)
_neighbors = st.one_of(
    _dates,
    _amounts,
    _numbers,
    st.sampled_from(["12/27", "01/2028", "12-27", "0128", "1227 123456", "0128 123456 45.00"]),
)


def _masked(card: str) -> str:
    return "****" + digits_of(card)[-4:]


# A 16-digit card is grouped 4-4-4-4, and that grouping outranks any longer
# stretch, so the line comes out exactly as expected whatever sits beside it.
@given(card=_cards, neighbors=st.lists(_neighbors, min_size=1, max_size=3))
def test_a_card_is_masked_and_what_sits_beside_it_kept(card: str, neighbors: list[str]) -> None:
    for line in (
        f"VISA {card} {' '.join(neighbors)}",
        f"{' '.join(neighbors)} {card}",
        f"Paid {card}\t{' '.join(neighbors)} ok",
    ):
        assert mask_card_numbers(line) == line.replace(card, _masked(card))


@given(first=_cards, second=_cards)
def test_two_cards_on_a_line_are_both_masked(first: str, second: str) -> None:
    for separator in (" ", "  Exp 12/27  ", " 2026-10-15 "):
        line = f"{first}{separator}{second}"
        assert mask_card_numbers(line) == f"{_masked(first)}{separator}{_masked(second)}"


@given(first=_dates, second=_dates)
def test_two_dates_joined_by_a_dash_are_never_masked(first: str, second: str) -> None:
    line = f"Period {first} - {second}"
    assert mask_card_numbers(line) == line


def test_a_long_run_with_a_digit_free_stretch_is_linear() -> None:
    line = "1 " + "-" * 10_000 + " 2"
    started = time.perf_counter()
    assert mask_card_numbers(line) == line
    assert time.perf_counter() - started < 0.2
