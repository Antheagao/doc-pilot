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

# The spaces a card number may be grouped with: space, tab, no-break space, thin
# space, and narrow no-break space.
_SPACES = " \t\u00a0\u2009\u202f"

# A maximal run: it starts and ends on a digit, so "$1,234.56" is "1" and
# "234.56", and "2026-10-15" is one run of eight digits.
_RUN_RE = re.compile(rf"\d(?:[{_SPACES}\-.]*\d)*")
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


# A group of a run (see _card_windows) that is a date or a decimal amount, so
# never part of a card number: two dates joined by " - " would otherwise read
# as one sixteen-digit run, and a card printed next to a date or an amount
# would hide inside a run that is too long to be a card.
# A group of a run (see _card_windows) that is a date or a decimal amount, so
# never part of a card number: two dates joined by " - " would otherwise read
# as one sixteen-digit run, and a card printed next to a date or an amount
# would hide inside a run that is too long to be a card.
_NOT_CARD_GROUP_RE = re.compile(
    r"\d{4}[-.]\d{1,2}[-.]\d{1,2}|\d{1,2}[-.]\d{1,2}[-.]\d{4}|\d+\.\d{2}"
)
_GROUP_RE = re.compile(rf"[^{_SPACES}]+")
# How card numbers are printed in groups: Visa and Mastercard 4-4-4-4 (up to 19
# digits as 4-4-4-4-3), Amex 4-6-5, Diners Club 4-6-4, and 6-13 for some Maestro
# and local cards. A number in one unbroken group is a card shape too.
_CARD_GROUPINGS = {(4, 4, 4, 4), (4, 4, 4, 4, 3), (4, 6, 5), (4, 6, 4), (6, 13)}


def _card_windows(run: str) -> list[tuple[int, int, str]]:
    """The (start, end, digits) of each card-shaped stretch of whole groups in
    a digit run, none overlapping. A group is what spaces separate ("4111",
    "12-27"); a stretch is 13 to 19 digits that pass the Luhn check. When
    stretches overlap, one grouped like a card (_CARD_GROUPINGS) wins over one
    that is not, then the longer, then the leftmost: so a card is taken whole
    when a number or an expiry sits beside it, and not as the tail of a chance
    match that takes in its neighbor. A date or amount group is never inside a
    stretch and splits the run."""
    groups = []
    for token in _GROUP_RE.finditer(run):
        digits = digits_of(token.group())
        if digits:
            groups.append((token.start(), token.end(), digits, _NOT_CARD_GROUP_RE.fullmatch(token.group())))
    found: list[tuple[bool, int, int, str]] = []
    for first in range(len(groups)):
        digits = ""
        lengths: list[int] = []
        for last in range(first, len(groups)):
            if groups[last][3]:
                break
            digits += groups[last][2]
            lengths.append(len(groups[last][2]))
            if len(digits) > CARD_MAX_DIGITS:
                break
            if len(digits) >= CARD_MIN_DIGITS and passes_luhn(digits):
                shaped = len(lengths) == 1 or tuple(lengths) in _CARD_GROUPINGS
                found.append((shaped, first, last, digits))
    found.sort(key=lambda window: (not window[0], -len(window[3]), window[1]))
    taken = [False] * len(groups)
    windows: list[tuple[int, int, str]] = []
    for _, first, last, digits in found:
        if not any(taken[first : last + 1]):
            taken[first : last + 1] = [True] * (last - first + 1)
            windows.append((groups[first][0], groups[last][1], digits))
    return sorted(windows)


def mask_card_numbers(text: str) -> str:
    """`text` with each card-shaped stretch of a digit run (13 to 19 digits
    that pass the Luhn check) replaced by "****" and its last four digits.
    Stricter than mask_digit_runs, for text that is not about payment at all
    (the quote of a total that happens to share a line with the card): an
    order reference of sixteen digits that is not a card number is left alone.

    A card is found inside a longer run too, so "VISA 4111 1111 1111 1111
    12-27" is masked and its expiry kept, and a date, an amount, or another
    number beside it does not hide it. Dates and amounts are never part of a
    card, so "2026-10-01 - 2026-10-31" is left alone. Known limits: a chance
    Luhn match that takes in the digits beside a card can show the wrong last
    four, and a reference number of 17 to 23 digits, grouped by spaces, can be
    partly masked when a stretch of it passes the check."""

    def replace(run: re.Match[str]) -> str:
        parts, cursor = [], 0
        for start, end, digits in _card_windows(run.group()):
            parts.append(run.group()[cursor:start])
            parts.append("****" + digits[-4:])
            cursor = end
        parts.append(run.group()[cursor:])
        return "".join(parts)

    return _RUN_RE.sub(replace, text)
