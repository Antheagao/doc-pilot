"""Split a page of text into retrievable chunks.

Pure functions, no I/O. The one invariant everything downstream relies on:
a chunk's `text` is always exactly `page_text[char_start:char_end]`. A
citation is those offsets into the stored page text, so the span a
search result cites is precisely the span that was retrieved -- never a
normalized or re-joined approximation of it.

Chunks are packed on line boundaries: receipts, invoices and forms are
line-structured (one line item per line), and cutting mid-line would
split a description from its price. A single line longer than
max_chars is the only case that gets cut mid-line, into fixed windows.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def _line_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every non-blank line in `text`, with each
    line's leading/trailing whitespace excluded from its span. Blank lines
    produce no span; they still count toward a chunk's character length
    when they fall between two packed lines, since the chunk is a
    contiguous substring.
    """
    spans = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            start = pos + (len(line) - len(line.lstrip()))
            spans.append((start, start + len(stripped)))
        pos += len(line)
    return spans


def _split_long_span(
    start: int, end: int, max_chars: int, overlap_chars: int
) -> list[tuple[int, int]]:
    """Fixed windows over one over-long line: max_chars wide, each
    starting overlap_chars before the previous one ended."""
    step = max_chars - overlap_chars
    windows = []
    window_start = start
    while True:
        window_end = min(window_start + max_chars, end)
        windows.append((window_start, window_end))
        if window_end >= end:
            return windows
        window_start += step


def chunk_page(
    text: str, page_number: int, *, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Chunk one page. max_chars <= 0 means no limit: the whole page (its
    first through last non-blank line) becomes a single chunk.

    Lines are packed greedily until adding the next one would make the
    chunk longer than max_chars. The next chunk then starts by repeating
    the previous chunk's trailing lines, as many as fit in overlap_chars,
    so an item whose meaning spans two adjacent lines is whole in at
    least one chunk. The start always advances by at least one line, so
    this terminates even when overlap_chars could hold every line.

    Returns [] for a blank page.
    """
    if max_chars > 0 and not 0 <= overlap_chars < max_chars:
        raise ValueError(
            f"overlap_chars must be in [0, max_chars); got {overlap_chars=} {max_chars=}"
        )

    spans = _line_spans(text)
    if not spans:
        return []

    if max_chars <= 0:
        windows = [(spans[0][0], spans[-1][1])]
    else:
        windows = []
        i = 0
        while i < len(spans):
            line_start, line_end = spans[i]
            if line_end - line_start > max_chars:
                windows.extend(_split_long_span(line_start, line_end, max_chars, overlap_chars))
                i += 1
                continue

            j = i
            while j + 1 < len(spans) and spans[j + 1][1] - line_start <= max_chars:
                j += 1
            windows.append((line_start, spans[j][1]))
            if j + 1 >= len(spans):
                break

            # Carry trailing lines of this chunk into the next one, up to
            # overlap_chars, but always move past line i.
            next_i = j + 1
            while (
                next_i - 1 > i
                and spans[j][1] - spans[next_i - 1][0] <= overlap_chars
                and spans[j + 1][1] - spans[next_i - 1][0] <= max_chars
            ):
                next_i -= 1
            i = next_i

    return [
        Chunk(
            page_number=page_number,
            chunk_index=index,
            text=text[start:end],
            char_start=start,
            char_end=end,
        )
        for index, (start, end) in enumerate(windows)
    ]


def document_title(first_page_text: str) -> str | None:
    """The first non-blank line of a document's first page -- the vendor
    name on a receipt, the title of a form -- used as the context header
    that tells a chunk cut from the middle of a page which document it
    belongs to. None for a blank first page.
    """
    spans = _line_spans(first_page_text)
    if not spans:
        return None
    start, end = spans[0]
    return first_page_text[start:end]


def context_header(title: str | None, page_number: int, page_count: int) -> str:
    """The search-only prefix for a chunk (DocumentChunk.context)."""
    where = f"page {page_number} of {page_count}"
    return f"{title} ({where})" if title else f"({where})"
