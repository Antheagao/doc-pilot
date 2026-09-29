from itertools import pairwise

import pytest

from app.retrieval.chunking import chunk_page, context_header, document_title

RECEIPT = """Northgate Office Outfitters
500 Commerce Dr, Columbus, OH

Date: 2025-09-01

Item                           Qty      Price      Total
Cedar Bird Feeder                4      $21.10     $84.40
Whiteboard Marker Set            2       $5.75     $11.50
AA Batteries (8-pack)            1       $6.80      $6.80
LED Desk Lamp                    2      $19.70     $39.40

Subtotal: $142.10
Tax: $11.72
Total: $153.82
"""


def _assert_offsets_exact(text: str, chunks) -> None:
    """The invariant citations depend on: every chunk IS a substring of
    the page at exactly its recorded offsets."""
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_whole_page_mode_is_one_chunk_spanning_first_to_last_line() -> None:
    chunks = chunk_page(RECEIPT, 1, max_chars=0, overlap_chars=0)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Northgate Office Outfitters")
    assert chunks[0].text.endswith("Total: $153.82")
    _assert_offsets_exact(RECEIPT, chunks)


@pytest.mark.parametrize("max_chars,overlap", [(200, 80), (120, 60), (80, 0), (61, 30)])
def test_chunks_respect_max_chars_and_keep_exact_offsets(max_chars: int, overlap: int) -> None:
    chunks = chunk_page(RECEIPT, 3, max_chars=max_chars, overlap_chars=overlap)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= max_chars for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.page_number == 3 for chunk in chunks)
    _assert_offsets_exact(RECEIPT, chunks)


def test_every_line_lands_in_some_chunk() -> None:
    chunks = chunk_page(RECEIPT, 1, max_chars=120, overlap_chars=60)
    covered = " ".join(chunk.text for chunk in chunks)
    for line in RECEIPT.splitlines():
        if line.strip():
            assert line.strip() in covered


def test_chunks_never_split_a_line_that_fits() -> None:
    """Packing is on line boundaries: each chunk starts at the start of a
    line and ends at the end of one."""
    lines = {line.strip() for line in RECEIPT.splitlines() if line.strip()}
    for chunk in chunk_page(RECEIPT, 1, max_chars=130, overlap_chars=50):
        chunk_lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        assert set(chunk_lines) <= lines


def test_overlap_repeats_trailing_lines_of_the_previous_chunk() -> None:
    chunks = chunk_page(RECEIPT, 1, max_chars=130, overlap_chars=70)

    for previous, current in pairwise(chunks):
        shared = RECEIPT[current.char_start : previous.char_end]
        assert shared, "consecutive chunks should overlap on this fixture"
        assert previous.text.endswith(shared)
        assert current.text.startswith(shared)
        assert len(shared) <= 70


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_page(RECEIPT, 1, max_chars=130, overlap_chars=0)

    for previous, current in pairwise(chunks):
        assert current.char_start >= previous.char_end


def test_overlap_budget_large_enough_for_whole_chunk_still_makes_progress() -> None:
    """overlap close to max_chars could otherwise re-emit the same chunk
    forever; each new chunk must start past the previous one's start."""
    text = "\n".join(f"line {i}" for i in range(40))
    chunks = chunk_page(text, 1, max_chars=30, overlap_chars=29)

    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(set(starts))
    assert chunks[-1].text.endswith("line 39")
    _assert_offsets_exact(text, chunks)


def test_single_overlong_line_is_split_into_overlapping_windows() -> None:
    long_line = "x" * 250
    text = f"header\n{long_line}\nfooter"
    chunks = chunk_page(text, 1, max_chars=100, overlap_chars=20)

    windows = [chunk for chunk in chunks if set(chunk.text) == {"x"}]
    assert [len(w.text) for w in windows] == [100, 100, 90]
    assert windows[1].char_start == windows[0].char_start + 80
    assert chunks[0].text == "header"
    assert chunks[-1].text == "footer"
    _assert_offsets_exact(text, chunks)


def test_blank_page_has_no_chunks() -> None:
    assert chunk_page("", 1, max_chars=200, overlap_chars=80) == []
    assert chunk_page("  \n\n \t\n", 1, max_chars=0, overlap_chars=0) == []


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_page(RECEIPT, 1, max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_page(RECEIPT, 1, max_chars=100, overlap_chars=-1)


def test_document_title_is_first_non_blank_line() -> None:
    assert document_title(RECEIPT) == "Northgate Office Outfitters"
    assert document_title("\n\n  RECEIPT  \nDate: x") == "RECEIPT"
    assert document_title("   \n") is None


def test_context_header_names_title_and_page() -> None:
    assert context_header("Northgate Office Outfitters", 2, 3) == (
        "Northgate Office Outfitters (page 2 of 3)"
    )
    assert context_header(None, 1, 1) == "(page 1 of 1)"
