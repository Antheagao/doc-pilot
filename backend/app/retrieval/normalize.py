"""Canonical forms for the tokens full-text search gets wrong on receipts.

Postgres's text parser splits amounts in the formats receipts actually
print: `27,82 EUR` becomes the lexemes '27' and '82', and `$1,234.56`
becomes '1' and '234.56' -- so a search for "27.82" or "1234.56" can never
match them. Currency markers don't line up with how people ask either:
the document says `EUR`, the question says "euros".

At index time, search_aliases() lists each chunk's amounts in one
canonical form (plain digits, dot decimal) plus a currency word per
currency marker; they go into the chunk's search-only `search_aliases`
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


def _canonical(text: str) -> list[str]:
    amounts = []
    for whole, cents in _EU_AMOUNT.findall(text):
        amounts.append(f"{whole.replace('.', '')}.{cents}")
    for whole, decimals in _US_THOUSANDS.findall(text):
        amounts.append(whole.replace(",", "") + decimals)
    return amounts


def search_aliases(text: str) -> str | None:
    """Space-separated canonical amounts and currency words for a chunk,
    or None when there is nothing the parser would have missed."""
    aliases = _canonical(text)
    aliases.extend(word for pattern, word in _CURRENCY_WORDS if pattern.search(text))
    return " ".join(dict.fromkeys(aliases)) or None


def normalize_query(query: str) -> str:
    """The query with its amounts rewritten canonically (27,82 -> 27.82,
    1,234.56 -> 1234.56), so they meet the aliases at index time."""
    query = _EU_AMOUNT.sub(lambda m: f"{m.group(1).replace('.', '')}.{m.group(2)}", query)
    query = _US_THOUSANDS.sub(lambda m: m.group(1).replace(",", "") + (m.group(2) or ""), query)
    return query
