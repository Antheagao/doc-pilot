"""Demo scenarios: one function per GIF, registered by slug.

A scenario drives the demo frontend through the Recording (harness.py) and
returns when the final state has been held. It uses the role, label and
text locators the frontend already exposes (plus the review card's class,
the one repeated block without a role); there are no test ids in the app
for recordings to depend on.

Scenarios run against `demo.py seed --reset` data and may change it (a
review correction resolves its field), so record after a fresh reset.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness import Recording
    from playwright.sync_api import Page

Scenario = Callable[["Page", "Recording"], None]

# Not a feature demo: proves the harness and encoder end to end. Its output
# stays in scripts/demos/out/ and never reaches screenshots/features/.
SMOKE_SLUG = "smoke"
SMOKE_DOCUMENT = "020-dense-electronics-order.png"


def smoke(page: Page, rec: Recording) -> None:
    """Home page, a short scroll of the document list, open one document."""
    document = page.get_by_role("link", name=re.compile(re.escape(SMOKE_DOCUMENT)))
    rec.open(rec.base_url, page.get_by_text("docs processed"), document)
    rec.beat(2)
    rec.scroll(200)
    rec.beat()
    rec.click(document)
    page.get_by_role("columnheader", name="Description").wait_for(state="visible")
    rec.beat(2)
    rec.end_hold()


# Hybrid search finds the desk lamp by meaning (no shared words), then
# keyword mode finds the Berlin receipt from a typed amount.
SEARCH_MEANING_QUERY = "a light for my workspace"
SEARCH_MEANING_HIT = "018-dense-office-outfitters.png"
SEARCH_KEYWORD_QUERY = "27.82 euros"
SEARCH_KEYWORD_HIT = "024-eur-bakery-berlin.png"


def search_hybrid(page: Page, rec: Recording) -> None:
    """A meaning query in hybrid mode, then an amount in keyword mode."""
    query = page.get_by_label("Search query")
    rec.open(f"{rec.base_url}/search", query)
    rec.beat()
    rec.type_text(query, SEARCH_MEANING_QUERY)
    page.keyboard.press("Enter")
    page.get_by_role("link", name=SEARCH_MEANING_HIT).first.wait_for(state="visible")
    rec.beat(2)
    rec.retype(query, SEARCH_KEYWORD_QUERY)
    rec.beat(0.5)
    rec.click(page.get_by_role("radio", name="Keywords"))
    page.get_by_role("link", name=SEARCH_KEYWORD_HIT).first.wait_for(state="visible")
    rec.beat(2)
    rec.end_hold()


# The staged misread (backend/app/demo_seed.py STAGED_REVIEW_CASES): the
# bakery note's subtotal reads 46.25; the document says 45.25.
REVIEW_DOCUMENT = "016-handwriting-bakery-note.png"
REVIEW_DOCUMENT_ID = "d03f3f24-6d81-5f98-8454-d7ca27dd5b3f"  # uuid5, stable across resets
REVIEW_CORRECT_VALUE = "45.25"


def review_correction(page: Page, rec: Recording) -> None:
    """The flagged subtotal on the document, corrected in the review queue,
    then the document again with the original struck through."""
    subtotal = page.get_by_text("46.25", exact=True)
    rec.open(f"{rec.base_url}/documents/{REVIEW_DOCUMENT_ID}", subtotal)
    rec.beat(1.5)
    rec.click(page.get_by_role("link", name=re.compile(r"^Review")))
    card = page.locator(".review-row").filter(
        has=page.get_by_role("link", name=REVIEW_DOCUMENT)
    )
    card.wait_for(state="visible")
    rec.beat()
    rec.click(card.get_by_role("button", name="Correct…"))
    rec.retype(card.get_by_placeholder("Corrected value"), REVIEW_CORRECT_VALUE)
    rec.beat(0.5)
    rec.click(card.get_by_role("button", name="Save correction"))
    card.wait_for(state="detached")
    rec.beat(1.5)
    # The document has no link back from the queue once its card is gone;
    # browser back returns to it, and the route change refreshes the badge.
    page.go_back()
    page.get_by_text("was 46.25").wait_for(state="visible")
    rec.beat(2)
    rec.end_hold()


SCENARIOS: dict[str, Scenario] = {
    SMOKE_SLUG: smoke,
    "search-hybrid": search_hybrid,
    "review-correction": review_correction,
}
