import pytest

from app.retrieval.normalize import normalize_query, search_aliases


@pytest.mark.parametrize(
    "text,aliases",
    [
        ("Total: 27,82 EUR", "27.82 euro"),
        ("Total: 1.234,56 EUR", "1234.56 euro"),
        ("Total: $1,234.56", "1234.56 dollar"),
        ("Total: $1,234", "1234 dollar"),
        ("Subtotal: 2,95 EUR  Total: 5,90 EUR", "2.95 5.90 euro"),
        ("Total: $425.58", "dollar"),  # already canonical; only the currency word
        ("Total: 12.00 GBP", "pound"),
    ],
)
def test_search_aliases_canonicalize_amounts_and_currencies(text, aliases) -> None:
    assert search_aliases(text) == aliases


@pytest.mark.parametrize(
    "text",
    ["Qty 2  Item 12, 3", "Date: 31/05/2026", "Date: 2025-09-01", "Notebook, ruled 80pg", ""],
)
def test_non_amounts_produce_no_aliases(text) -> None:
    assert search_aliases(text) is None


@pytest.mark.parametrize(
    "query,normalized",
    [
        ("the receipt for 27,82 euros", "the receipt for 27.82 euros"),
        ("invoice of $1,234.56", "invoice of $1234.56"),
        ("1.234,56 EUR total", "1234.56 EUR total"),
        ("total 425.58", "total 425.58"),
        ("12, 3 items", "12, 3 items"),
    ],
)
def test_normalize_query_matches_index_time_forms(query, normalized) -> None:
    assert normalize_query(query) == normalized


@pytest.mark.parametrize(
    "text,aliases",
    [
        ("Ironclad Hardware Supply\n78 Foundry Ave, Detroit, MI\n", "michigan"),
        ("Sunny Acres Farm Stand\n40 Rural Route 2, Ithaca, NY 14850", "new york"),
        ("Ridgeline Bike Works\n88 Summit Ave, Bend, OR 97701-1234\nTotal: $35.60", "dollar oregon"),
        ("Lindenplatz Bakery\n12 Marktplatz, Berlin, Germany", None),
        # Not an address line: no comma, lowercase, mid-line, unknown code.
        ("Cash OR card accepted", None),
        ("Pay by cash, or card", None),
        ("Code: MI 4471, then total", None),
        ("Reno, ZZ", None),
    ],
)
def test_us_address_lines_get_their_state_name(text, aliases) -> None:
    assert search_aliases(text) == aliases
