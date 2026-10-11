"""Demo scenarios: one function per GIF, registered by slug.

A scenario drives the demo frontend through the Recording (harness.py) and
returns when the final state has been held. It uses the role, label and
text locators the frontend already exposes (plus the review card's class,
the one repeated block without a role); there are no test ids in the app
for recordings to depend on.

Scenarios run against `demo.py seed --reset` data and may change it (a
review correction resolves its field), so record after a fresh reset.

LIVE_SLUGS call the real model (the agent behind /ask and the document
chat) and spend money against the stack's DAILY_BUDGET_USD. demo.py checks
the key and the budget before such a take and reports the spend after it.
Their runs are stored, and the app shows them again (the Ask page's recent
questions, the document page's restored conversation), so a take needs a
fresh `demo.py seed --reset` and records `ask-agent` before `document-chat`
(`record all` keeps that order); a scenario that finds an earlier run on
the page fails before it spends anything.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness import Recording
    from playwright.sync_api import Locator, Page

Scenario = Callable[["Page", "Recording"], None]


class ScenarioError(Exception):
    """A take that cannot be used (for a live run: the app refused it, or
    the answer lacks what the GIF needs); the message says why."""


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


# The agent answers live, so a wait is anywhere from a few seconds to
# half a minute; the GIF shows it sped up, with the speed chip on screen.
LIVE_TIMEOUT_MS = 90_000
LIVE_SPEED = 3


def wait_for_live_answer(page: Page, rec: Recording, answer: Locator) -> None:
    """Fast-forward through the live wait until `answer` shows, and fail the
    take if the app shows an error banner instead (or beside it: a refused
    or truncated answer is not a take worth keeping). The 429 of an
    exhausted daily budget lands here: the page says `Not right now: ...`."""
    banner = page.locator(".error-banner")
    with rec.fast_forward(LIVE_SPEED):
        answer.or_(banner).first.wait_for(state="visible", timeout=LIVE_TIMEOUT_MS)
    if banner.count():
        raise ScenarioError(
            f"the live run did not answer cleanly: {banner.first.inner_text()}"
        )


# One answer, one document: the only receipt that came to $425.58.
ASK_QUESTION = "Which store's receipt came to $425.58?"
ASK_SOURCE = "018-dense-office-outfitters.png"


def ask_agent(page: Page, rec: Recording) -> None:
    """A live /ask run: the agent's steps as they stream (sped up), the
    cited answer, then the cited document."""
    rec.open(f"{rec.base_url}/ask", page.get_by_label("Question"))
    if page.locator(".history-row").count():
        raise ScenarioError(
            "the Ask page lists earlier questions: run `demo.py seed --reset` first"
        )
    rec.beat()
    # The class, not the role: a retake finds the earlier takes under
    # "Recent questions", whose entries are buttons with the same name.
    rec.click(page.locator("button.example-chip").filter(has_text=ASK_QUESTION))
    wait_for_live_answer(page, rec, page.locator(".answer-card .answer-text"))
    source = page.get_by_role("link", name=ASK_SOURCE).first
    if source.count() == 0:
        raise ScenarioError(f"the answer does not cite {ASK_SOURCE}")
    rec.beat(2)
    rec.click(source)
    page.locator(".field-row").first.wait_for(state="visible")
    rec.beat(2)
    rec.end_hold()


# The same document as the search demo's meaning hit (018), looked up by
# its stable uuid5 id; the suggestion chip is the first the panel offers.
CHAT_DOCUMENT_ID = "9a1df399-c746-56e8-b4bf-490ca9aa9be4"
CHAT_SUGGESTION = "What's the total?"


def document_chat(page: Page, rec: Recording) -> None:
    """A live per-document chat turn: a suggested question, then the
    answer's extracted-field citation selected so its row lights up."""
    suggestion = page.get_by_role("button", name=CHAT_SUGGESTION)
    rec.open(
        f"{rec.base_url}/documents/{CHAT_DOCUMENT_ID}",
        page.get_by_role("heading", name="Ask about this document"),
    )
    if suggestion.count() == 0:
        raise ScenarioError(
            "an earlier conversation about this document is shown: "
            "run `demo.py seed --reset` first"
        )
    rec.beat()
    rec.click(suggestion)
    # The suggestions give way to the live steps right under the cursor.
    rec.park()
    wait_for_live_answer(page, rec, page.locator(".chat-assistant .answer-text"))
    # The source chips, not the answer's numbered markers (same names).
    field_sources = page.locator("button.chat-source").filter(has_text="extracted field")
    if field_sources.count() == 0:
        raise ScenarioError("the answer cites no extracted field")
    rec.beat(2)
    rec.click(field_sources.first)
    page.locator(".field-row.cited, .line-items-table tr.cited").first.wait_for(
        state="visible"
    )
    rec.beat(2)
    rec.end_hold()


SCENARIOS: dict[str, Scenario] = {
    SMOKE_SLUG: smoke,
    "search-hybrid": search_hybrid,
    "review-correction": review_correction,
    "ask-agent": ask_agent,
    "document-chat": document_chat,
}
LIVE_SLUGS = frozenset({"ask-agent", "document-chat"})
