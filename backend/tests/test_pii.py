"""The digit-run rules of app/pii.py: pure functions, no database."""

import pytest

from app.pii import has_long_digit_run, mask_card_numbers, mask_digit_runs, passes_luhn


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
        ("4111 1111 1111 1111 1111 1111", "4111 1111 1111 1111 1111 1111"),
        ("Paid 2026-10-15 $1,234.56", "Paid 2026-10-15 $1,234.56"),
    ],
)
def test_mask_card_numbers(text: str, masked: str) -> None:
    assert mask_card_numbers(text) == masked
