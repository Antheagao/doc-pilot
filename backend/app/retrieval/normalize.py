"""Canonical forms for the tokens full-text search gets wrong on receipts.

Postgres's text parser splits amounts in the formats receipts actually
print: `27,82 EUR` becomes the lexemes '27' and '82', and `$1,234.56`
becomes '1' and '234.56' -- so a search for "27.82" or "1234.56" can never
match them. Currency markers don't line up with how people ask either:
the document says `EUR`, the question says "euros". Addresses have the
same problem: a receipt prints `Detroit, MI`, the question asks about
"stores in Michigan".

At index time, search_aliases() lists each chunk's amounts in one
canonical form (plain digits, dot decimal), a currency word per currency
marker, and the state name for a US `City, ST` address line; they go
into the chunk's search-only `search_aliases`
column, which the generated tsvector includes. normalize_query() rewrites
amounts in a query into the same form. The cited text is never touched.
"""

import re

# 1.234,56 / 27,82 -- dot (or no) thousands, comma decimal (EU style).
_EU_AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+),(\d{2})(?![\d,])")
# 1,234.56 / 1,234 -- comma thousands (US style), optional dot decimal.
_US_THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(\.\d{2})?(?![\d,])")

_CURRENCY_WORDS = (
    (re.compile(r"€|\bEUR\b|\beuros?\b", re.IGNORECASE), "euro"),
    (re.compile(r"\$|\bUSD\b|\bdollars?\b", re.IGNORECASE), "dollar"),
    (re.compile(r"£|\bGBP\b|\bpounds?\b", re.IGNORECASE), "pound"),
)


_US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "district of columbia",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon",
    "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
}
# "Detroit, MI" / "Bend, OR 97701" at the end of a line: the postal code
# only counts after a comma, in capitals, closing the line (optionally
# with a ZIP) -- so "OR", "IN" and "ME" in running text never match.
_CITY_STATE = re.compile(r",[ \t]*([A-Z]{2})(?:[ \t]+\d{5}(?:-\d{4})?)?[ \t]*$", re.MULTILINE)


def _canonical(text: str) -> list[str]:
    amounts = []
    for whole, cents in _EU_AMOUNT.findall(text):
        amounts.append(f"{whole.replace('.', '')}.{cents}")
    for whole, decimals in _US_THOUSANDS.findall(text):
        amounts.append(whole.replace(",", "") + decimals)
    return amounts


def search_aliases(text: str) -> str | None:
    """Space-separated canonical amounts, currency words and state names
    for a chunk, or None when there is nothing the parser would have
    missed."""
    aliases = _canonical(text)
    aliases.extend(word for pattern, word in _CURRENCY_WORDS if pattern.search(text))
    aliases.extend(_US_STATES[code] for code in _CITY_STATE.findall(text) if code in _US_STATES)
    return " ".join(dict.fromkeys(aliases)) or None


def normalize_query(query: str) -> str:
    """The query with its amounts rewritten canonically (27,82 -> 27.82,
    1,234.56 -> 1234.56), so they meet the aliases at index time."""
    query = _EU_AMOUNT.sub(lambda m: f"{m.group(1).replace('.', '')}.{m.group(2)}", query)
    query = _US_THOUSANDS.sub(lambda m: m.group(1).replace(",", "") + (m.group(2) or ""), query)
    return query
