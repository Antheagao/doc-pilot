"""Demo scenarios: one function per GIF, registered by slug.

A scenario drives the demo frontend through the Recording (harness.py) and
returns when the final state has been held. It only uses role, label and
text locators the frontend already exposes; there are no test ids in the
app for recordings to depend on.

Later demo tasks add their scenarios here beside `smoke`.
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


SCENARIOS: dict[str, Scenario] = {SMOKE_SLUG: smoke}
