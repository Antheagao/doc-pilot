"""Recording harness: drive the demo frontend in Chromium and record it.

Playwright is an optional extra (`pip install -e ".[dev,demo]"`, then
`python -m playwright install chromium`), imported lazily inside
recording() so everything else in scripts/demos imports without it.

A scenario gets a Recording (rec): rec.page is the Playwright page, and
rec.click / rec.type_text / rec.scroll / rec.beat give it human pacing.
Every pointer move is a smooth glide with a visible cursor (the overlay
below), because a headless recording has no real cursor. Time compression
is never hidden: `with rec.fast_forward(4):` speeds that stretch up in the
encode and shows an "4x" chip on the page while it is active.

The recording is the page.screencast frame stream, saved as JPEG files
(quality 100) with their timestamps, not Playwright's WebM: that video is
VP8 encoded in real time, and its encoder noise on every text edge makes
every GIF frame differ, so no scenario fits under the 1 MB GIF limit.
Chromium only sends a frame when the page changes, so a still screen costs
nothing in the GIF. The timestamps are the same clock as ours, so the
times in the sidecar line up with the frames exactly.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from encode import Segment, Sidecar, frame_list

if TYPE_CHECKING:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Locator,
        Page,
        ScreencastFrame,
    )

OUT_DIR = Path(__file__).resolve().parent / "out"

VIEWPORT = {"width": 1280, "height": 800}

# Pacing: tuned so a scenario reads at a human speed in a 12 fps GIF.
BEAT = 0.7  # seconds of stillness between actions
TYPE_DELAY_MS = 80  # per keystroke
MOVE_STEPS = 30  # mouse moves per glide to a target
GLIDE_MS = 600  # how long a glide takes
SCROLL_STEP_PX = 24  # per wheel tick of Recording.scroll
SCROLL_TICK_MS = 16
END_HOLD = 1.5  # seconds on the final state before the video ends

# Seconds added to the measured first-paint time: a frame needs a moment
# to reach the screen.
LEAD_IN_PAD = 0.15
FRAME_QUALITY = 100  # JPEG quality of the screencast frames
START_POINTER = (1224, 340)  # rests in the right gutter, off the content

# Injected into every document before its scripts run. It draws the cursor
# arrow, pulses a ring on mousedown, and shows the speed chip. The pointer
# position and the chip label live in sessionStorage so a full page load
# (not just a client-side route change) keeps both on screen.
OVERLAY_SCRIPT = """
(() => {
  const KEY_POS = '__recPointer';
  const KEY_CHIP = '__recSpeed';
  const read = (key) => { try { return sessionStorage.getItem(key); } catch { return null; } };
  const write = (key, value) => {
    try {
      if (value === null) sessionStorage.removeItem(key);
      else sessionStorage.setItem(key, value);
    } catch {}
  };

  const install = () => {
    const root = document.documentElement;
    const host = document.createElement('div');
    host.setAttribute('aria-hidden', 'true');
    host.style.cssText =
      'position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden;';
    host.innerHTML = `
      <svg id="__recCursor" width="28" height="28" viewBox="0 0 24 24"
           style="position:absolute;left:0;top:0;display:none;filter:drop-shadow(0 1px 2px rgba(0,0,0,.45));">
        <path d="M3 2 L3 19 L7.6 14.8 L10.6 21.4 L13.4 20.2 L10.5 13.7 L16.8 13.4 Z"
              fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/>
      </svg>
      <div id="__recChip"
           style="position:absolute;right:16px;bottom:16px;display:none;padding:4px 12px;
                  border-radius:999px;background:rgba(17,17,17,.82);color:#fff;
                  font:600 15px/1.4 system-ui,sans-serif;letter-spacing:.02em;"></div>`;
    root.appendChild(host);
    const cursor = host.querySelector('#__recCursor');
    const chip = host.querySelector('#__recChip');

    const place = (x, y) => {
      cursor.style.display = 'block';
      cursor.style.transform = `translate(${x - 3.5}px, ${y - 2.3}px)`;
    };
    const showChip = (label) => {
      chip.textContent = label || '';
      chip.style.display = label ? 'block' : 'none';
    };
    window.__recSetChip = (label) => { write(KEY_CHIP, label); showChip(label); };

    const saved = read(KEY_POS);
    if (saved) { const [x, y] = saved.split(',').map(Number); place(x, y); }
    showChip(read(KEY_CHIP));

    window.addEventListener('mousemove', (e) => {
      place(e.clientX, e.clientY);
      write(KEY_POS, `${e.clientX},${e.clientY}`);
    }, true);
    window.addEventListener('mousedown', (e) => {
      const ring = document.createElement('div');
      ring.style.cssText =
        `position:absolute;left:${e.clientX - 20}px;top:${e.clientY - 20}px;width:40px;height:40px;` +
        'box-sizing:border-box;border-radius:50%;border:3px solid rgba(37,99,235,.9);' +
        'background:rgba(37,99,235,.18);';
      host.appendChild(ring);
      ring.animate(
        [{ transform: 'scale(.4)', opacity: 1 }, { transform: 'scale(1.5)', opacity: 0 }],
        { duration: 450, easing: 'ease-out' }
      ).onfinish = () => ring.remove();
    }, true);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', install, { once: true });
  } else {
    install();
  }
})();
"""


def new_context(browser: Browser) -> BrowserContext:
    """A context whose look is identical on every machine and run: fixed
    viewport, light theme, UTC, English, no device scaling."""
    context = browser.new_context(
        viewport=VIEWPORT,
        color_scheme="light",
        device_scale_factor=1,
        timezone_id="UTC",
        locale="en-US",
    )
    context.add_init_script(OVERLAY_SCRIPT)
    return context


class Recording:
    """A scenario's handle on the page being recorded."""

    def __init__(
        self,
        page: Page,
        slug: str,
        base_url: str,
        frames_dir: Path,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.page = page
        self.slug = slug
        self.base_url = base_url
        self._frames_dir = frames_dir
        self._clock = clock
        self._origin = clock()
        self._lead_in: float | None = None
        self._segments: list[Segment] = []
        self._end: float | None = None
        self._pointer = START_POINTER
        # (epoch seconds, file name) of each screencast frame, in order.
        self._frames: list[tuple[float, str]] = []

    def now(self) -> float:
        """Seconds since the recording started."""
        return self._clock() - self._origin

    def capture(self, frame: ScreencastFrame) -> None:
        """Save one screencast frame (the on_frame callback)."""
        name = f"{len(self._frames):06d}.jpg"
        (self._frames_dir / name).write_bytes(frame["data"])
        self._frames.append((frame["timestamp"] / 1000, name))

    def open(self, url: str, *ready: Locator) -> None:
        """Load url, wait until every `ready` locator is visible and the
        network is idle (no content left to pop in and shift the layout).
        Everything before that moment is the blank lead-in the encode trims."""
        self.page.goto(url)
        for locator in ready:
            locator.wait_for(state="visible")
        self.page.wait_for_load_state("networkidle")
        self.page.mouse.move(*START_POINTER)
        self.page.wait_for_timeout(100)
        self._lead_in = self.now() + LEAD_IN_PAD

    def beat(self, count: float = 1) -> None:
        """Stand still for `count` beats."""
        self.page.wait_for_timeout(count * BEAT * 1000)

    def end_hold(self) -> None:
        """Hold the final state so the loop does not cut off mid-thought."""
        self.page.wait_for_timeout(END_HOLD * 1000)

    def glide(self, locator: Locator) -> None:
        """Scroll the element into view and move the cursor to its center
        along an eased path, so the motion reads on a 12 fps GIF (a plain
        mouse.move(steps=N) jumps there in a few milliseconds)."""
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        if box is None:
            raise RuntimeError(f"{locator} has no bounding box to glide to")
        start_x, start_y = self._pointer
        end_x = box["x"] + box["width"] / 2
        end_y = box["y"] + box["height"] / 2
        for step in range(1, MOVE_STEPS + 1):
            progress = step / MOVE_STEPS
            eased = progress * progress * (3 - 2 * progress)
            self.page.mouse.move(
                start_x + (end_x - start_x) * eased,
                start_y + (end_y - start_y) * eased,
            )
            self.page.wait_for_timeout(GLIDE_MS / MOVE_STEPS)
        self._pointer = (end_x, end_y)

    def click(self, locator: Locator) -> None:
        """Glide to the element, pause briefly, click it."""
        self.glide(locator)
        self.page.wait_for_timeout(250)
        locator.click()

    def type_text(self, locator: Locator, text: str) -> None:
        """Click the field, then type at human speed."""
        self.click(locator)
        self.page.keyboard.type(text, delay=TYPE_DELAY_MS)

    def scroll(self, pixels: int) -> None:
        """Wheel-scroll the page by `pixels` (negative scrolls up) in small
        ticks, so the motion is smooth on a 12 fps GIF."""
        direction = 1 if pixels > 0 else -1
        remaining = abs(pixels)
        while remaining > 0:
            step = min(SCROLL_STEP_PX, remaining)
            self.page.mouse.wheel(0, direction * step)
            self.page.wait_for_timeout(SCROLL_TICK_MS)
            remaining -= step

    @contextmanager
    def fast_forward(self, factor: float) -> Iterator[None]:
        """Play the enclosed stretch `factor` times faster in the encode,
        with a "<factor>x" chip on the page for as long as it lasts."""
        self.page.evaluate("label => window.__recSetChip(label)", f"{factor:g}x")
        start = self.now()
        try:
            yield
        finally:
            end = self.now()
            self._segments.append(Segment(start, end, factor))
            self.page.evaluate("() => window.__recSetChip(null)")

    def stop(self) -> None:
        """End the recording: the moment of the last frame of the GIF."""
        self._end = self.now()
        self.page.screencast.stop()

    def _shift(self) -> float:
        """Seconds from our origin to the first frame, which is time 0 of
        the sidecar and of the frame list."""
        return self._frames[0][0] - self._origin

    def sidecar(self) -> Sidecar:
        """The sidecar for what was recorded; call after stop()."""
        if self._lead_in is None:
            raise RuntimeError("the scenario never called rec.open()")
        if self._end is None or not self._frames:
            raise RuntimeError("the recording was never stopped or has no frames")
        shift = self._shift()
        sidecar = Sidecar(
            slug=self.slug,
            lead_in=self._lead_in - shift,
            duration=self._end - shift,
            segments=tuple(
                Segment(s.start - shift, s.end - shift, s.factor)
                for s in self._segments
            ),
        )
        sidecar.validate()
        return sidecar

    def frame_list(self) -> str:
        """The concat list that replays the saved frames at their times."""
        if self._end is None or not self._frames:
            raise RuntimeError("the recording was never stopped or has no frames")
        first = self._frames[0][0]
        return frame_list(
            [(timestamp - first, name) for timestamp, name in self._frames],
            self._end - self._shift(),
        )


@contextmanager
def recording(slug: str, base_url: str, out_dir: Path = OUT_DIR) -> Iterator[Recording]:
    """Launch Chromium and record the `with` body. On exit the frames are in
    <out_dir>/<slug>.frames/ with their list in frames.txt, and the sidecar
    is <out_dir>/<slug>.json."""
    from playwright.sync_api import sync_playwright

    frames_dir = out_dir / f"{slug}.frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = new_context(browser)
            try:
                page = context.new_page()
                rec = Recording(page, slug, base_url, frames_dir)
                page.screencast.start(
                    on_frame=rec.capture, quality=FRAME_QUALITY, size=VIEWPORT
                )
                yield rec
                rec.stop()
            finally:
                context.close()
        finally:
            browser.close()
    (frames_dir / "frames.txt").write_text(rec.frame_list(), encoding="utf-8")
    (out_dir / f"{slug}.json").write_text(rec.sidecar().to_json(), encoding="utf-8")
