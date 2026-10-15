"""Does the number in a value's quote equal the value?

An extraction that carries evidence gives every amount the verbatim text it was
read from (`{page, quote}`). A model that writes a number it did not read tends
to leave a quote that shows a different one, so the check is: the value must
appear among the numbers the quote prints. It is deterministic, pure, and runs
in `Decimal`; app/reconcile.py turns each verdict into a `quote:<path>` check.

Strict on digits, lenient on format: the comparison is exact (46.25 never
agrees with 46.52 or 46.2), but any way of printing the number counts, and any
reading of an ambiguous one (`app.money.printed_amounts`). Signs and
parentheses are layout (`-$5.00` and `(5.00)` both print a discount of 5.00), so
the comparison is between magnitudes. A zero with a quote that prints no number
("Shipping: FREE", "Tax: -") is skipped, not failed: that is how a document says
zero. A quote that holds several numbers
(`2 x 4.50 = 9.00`) agrees with a value equal to any of them; it is the value
being absent from the quote that fails, not the quote holding more.

What it cannot see: a quote the model also got wrong, or invented to match. That
is a model reading a document twice badly, not an arithmetic fact, and the
reconciliation rules and the human review queue are the other defenses.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.money import printed_amounts, to_api_string, to_plain_string

STATUS_PASS = "pass"
STATUS_FAIL_MISMATCH = "fail_mismatch"
STATUS_FAIL_NO_NUMBER = "fail_no_number"
STATUS_SKIPPED_NO_QUOTE = "skipped_no_quote"
STATUS_SKIPPED_ZERO_UNPRINTED = "skipped_zero_unprinted"


def is_skipped(verdict: "QuoteVerdict") -> bool:
    """True for a verdict that judged nothing (no quote, or a zero said in words)."""
    return verdict.status.startswith("skipped_")


@dataclass(frozen=True)
class QuoteVerdict:
    """The outcome for one value and its quote. `numbers` are the readings the
    quote's text gives (empty when there is no quote or no number in it);
    `detail` is a sentence showing the value and the quote's numbers, fit for a
    person."""

    status: str
    detail: str
    numbers: tuple[Decimal, ...] = ()


def quote_agreement(
    value: Decimal,
    quote: str | None,
    *,
    places: int | None = None,
    minor_units: int | None = None,
) -> QuoteVerdict:
    """Compare a value with the numbers its quote prints.

    `skipped_no_quote` when the quote is missing or blank,
    `skipped_zero_unprinted` when the value is 0 and the quote prints no number,
    `fail_no_number` when it prints no number for any other value,
    `fail_mismatch` when none equals the value's magnitude, else `pass`.
    `places` is how many decimal places the field allows, which settles how an
    ambiguous `1,000` is read (see `app.money.printed_amounts`). `minor_units`
    pads the amounts in the detail sentence to a currency's places (`46.20`);
    without it they print as plain numbers (a quantity, a rate)."""
    if quote is None or not quote.strip():
        return QuoteVerdict(STATUS_SKIPPED_NO_QUOTE, "The field has no quote to check.")

    def show(amount: Decimal) -> str:
        return to_plain_string(amount) if minor_units is None else to_api_string(amount, minor_units)

    numbers = printed_amounts(quote, places)
    if not numbers and value == 0:
        return QuoteVerdict(
            STATUS_SKIPPED_ZERO_UNPRINTED, "The value is 0 and its quote prints no number."
        )
    if not numbers:
        return QuoteVerdict(
            STATUS_FAIL_NO_NUMBER,
            f"The value is {show(value)}, but its quote contains no number.",
        )
    if abs(value) in numbers:
        return QuoteVerdict(
            STATUS_PASS, f"The quote shows {show(abs(value))}, which is the value.", numbers
        )
    shown = ", ".join(show(number) for number in numbers)
    if len(numbers) == 1:
        detail = f"The value is {show(value)}, but its quote shows {shown}."
    else:
        detail = f"The value is {show(value)}, but none of the numbers in its quote ({shown}) is it."
    return QuoteVerdict(STATUS_FAIL_MISMATCH, detail, numbers)
