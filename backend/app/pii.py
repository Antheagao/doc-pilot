"""The card-number rule: no more than the last four digits of a card or account
number are ever stored.

Three places need the same idea of a digit run: the extraction masks what the
model returned (app/extraction.py, mask_card_data), a reviewer's correction is
refused when it holds one (app/records.py, normalize_correction), and both read
it from here. This module imports nothing from the app, so they can load it
without a cycle.

A digit run is a maximal stretch of digits that spaces, tabs, no-break spaces,
dashes, and dots may separate (how a card number is printed in groups). Digits
are matched in any script, so fullwidth digits do not slip past.
"""

import re
import unicodedata

# A maximal run: it starts and ends on a digit, so "$1,234.56" is "1" and
# "234.56", and "2026-10-15" is one run of eight digits.
_RUN_RE = re.compile(r"\d(?:[ \t \-.]*\d)*")
_FIVE_IN_A_ROW_RE = re.compile(r"\d{5}")

# A run is a card or account number when it holds this many digits in all, or
# five in a row: that keeps dates, amounts, and phone numbers readable and
# still catches a number printed without groups.
LONG_RUN_DIGITS = 13
# The lengths of a payment card number, for the Luhn sweep over quotes.
CARD_MIN_DIGITS = 13
CARD_MAX_DIGITS = 19

# What a stored payment_card_last4 looks like: exactly four ASCII digits.
CARD_LAST4_RE = re.compile(r"[0-9]{4}", re.ASCII)


def digits_of(text: str) -> str:
    """The digits of `text` as ASCII, in order (a fullwidth digit is folded)."""
    return "".join(str(unicodedata.digit(c)) for c in text if c.isdecimal())


def _is_long_run(run: str) -> bool:
    return len(digits_of(run)) >= LONG_RUN_DIGITS or _FIVE_IN_A_ROW_RE.search(run) is not None


def has_long_digit_run(text: str) -> bool:
    """True when `text` holds a digit run that looks like a card or account
    number: thirteen digits in all, or five in a row."""
    return any(_is_long_run(run.group()) for run in _RUN_RE.finditer(text))


def _masked(run: str) -> str:
    return "****" + digits_of(run)[-4:]


def mask_digit_runs(text: str) -> str:
    """`text` with each long digit run (see has_long_digit_run) replaced by
    "****" and its last four digits ("VISA 4111 1111 1111 1111" becomes
    "VISA ****1111"). Dates, amounts, and short numbers are left alone."""
    return _RUN_RE.sub(lambda run: _masked(run.group()) if _is_long_run(run.group()) else run.group(), text)


def passes_luhn(digits: str) -> bool:
    """The Luhn checksum every payment card number satisfies."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def mask_card_numbers(text: str) -> str:
    """`text` with each run that is card-shaped (13 to 19 digits that pass the
    Luhn check) replaced by "****" and its last four digits. Stricter than
    mask_digit_runs, for text that is not about payment at all (the quote of a
    total that happens to share a line with the card): an order reference of
    sixteen digits that is not a card number is left alone."""

    def replace(run: re.Match[str]) -> str:
        digits = digits_of(run.group())
        if CARD_MIN_DIGITS <= len(digits) <= CARD_MAX_DIGITS and passes_luhn(digits):
            return _masked(run.group())
        return run.group()

    return _RUN_RE.sub(replace, text)
