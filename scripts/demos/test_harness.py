"""Tests for the recording harness's timing, with a fake page and clock.

Playwright is never imported (run: python -m pytest ../scripts/demos -q).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import harness
import pytest
from encode import Segment


class FakePage:
    """Records the calls the harness makes and lets the clock advance."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.evaluated: list[tuple[str, object]] = []
        self.moves: list[tuple[float, float]] = []

    def evaluate(self, script: str, arg: object = None) -> None:
        self.evaluated.append((script, arg))

    def wait_for_timeout(self, ms: float) -> None:
        self.clock.now += ms / 1000

    def goto(self, url: str) -> None:
        self.clock.now += 1

    def wait_for_load_state(self, state: str) -> None:
        pass

    @property
    def mouse(self) -> FakeMouse:
        return FakeMouse(self.moves)


class FakeMouse:
    def __init__(self, moves: list[tuple[float, float]]) -> None:
        self.moves = moves

    def move(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    def wheel(self, dx: float, dy: float) -> None:
        self.moves.append((dx, dy))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(tmp_path: Path) -> tuple[harness.Recording, FakePage, FakeClock]:
    clock = FakeClock()
    page = FakePage(clock)
    rec = harness.Recording(page, "demo", "http://x", tmp_path, clock=clock)  # type: ignore[arg-type]
    return rec, page, clock


def frame(epoch_seconds: float) -> dict:
    return {"data": b"jpeg", "timestamp": epoch_seconds * 1000}


def test_fast_forward_records_its_span_and_shows_then_clears_the_chip(
    tmp_path: Path,
) -> None:
    rec, page, clock = make(tmp_path)
    clock.now += 2
    with rec.fast_forward(4):
        clock.now += 3
    assert rec._segments == [Segment(2.0, 5.0, 4)]
    assert page.evaluated[0][1] == "4x"
    assert "null" in page.evaluated[1][0]


def test_fast_forward_still_records_when_the_body_raises(tmp_path: Path) -> None:
    rec, page, clock = make(tmp_path)
    with pytest.raises(RuntimeError), rec.fast_forward(8):
        clock.now += 1
        raise RuntimeError("boom")
    assert rec._segments == [Segment(0.0, 1.0, 8)]
    assert "null" in page.evaluated[-1][0]


def test_sidecar_times_are_relative_to_the_first_frame(tmp_path: Path) -> None:
    rec, _page, clock = make(tmp_path)
    rec.capture(frame(1000.25))  # the first frame arrives 0.25 s after the origin
    rec._lead_in = 1.25
    clock.now += 2.25
    with rec.fast_forward(4):
        clock.now += 2
    clock.now += 1
    rec._end = rec.now()
    sidecar = rec.sidecar()
    assert sidecar.lead_in == pytest.approx(1.0)
    assert sidecar.duration == pytest.approx(5.0)
    assert sidecar.segments == (Segment(2.0, 4.0, 4),)


def test_frame_list_uses_the_same_zero_as_the_sidecar(tmp_path: Path) -> None:
    rec, _page, clock = make(tmp_path)
    rec.capture(frame(1000.25))
    rec.capture(frame(1001.25))
    clock.now += 3.25
    rec._end = rec.now()
    assert rec.frame_list() == (
        "file '000000.jpg'\noption framerate 1000\nduration 1.000\n"
        "file '000001.jpg'\noption framerate 1000\nduration 2.000\n"
        "file '000001.jpg'\noption framerate 1000\n"
    )
    assert (tmp_path / "000001.jpg").read_bytes() == b"jpeg"


def test_a_scenario_that_never_opened_a_page_has_no_sidecar(tmp_path: Path) -> None:
    rec, _page, _clock = make(tmp_path)
    rec.capture(frame(1000.0))
    rec._end = 1.0
    with pytest.raises(RuntimeError, match="rec.open"):
        rec.sidecar()


def test_the_overlay_pins_the_look_of_every_recording() -> None:
    assert harness.VIEWPORT == {"width": 1280, "height": 800}
    assert "pointer-events:none" in harness.OVERLAY_SCRIPT
    assert "2147483647" in harness.OVERLAY_SCRIPT


def test_a_second_open_keeps_the_first_lead_in_and_resets_the_pointer(
    tmp_path: Path,
) -> None:
    rec, _page, clock = make(tmp_path)
    rec.open("http://x/")
    first_lead_in = rec._lead_in
    rec._pointer = (10, 10)
    clock.now += 5
    rec.open("http://x/review")
    assert rec._lead_in == first_lead_in
    assert rec._pointer == harness.START_POINTER


def test_park_glides_the_pointer_back_to_its_resting_spot(tmp_path: Path) -> None:
    rec, page, _ = make(tmp_path)
    rec._pointer = (100.0, 200.0)
    rec.park()
    assert len(page.moves) == harness.MOVE_STEPS
    assert page.moves[-1] == pytest.approx(harness.START_POINTER)
    assert rec._pointer == harness.START_POINTER


class FakeLocator:
    def __init__(self, box: dict[str, float] | None) -> None:
        self.box = box

    def scroll_into_view_if_needed(self) -> None:
        pass

    def bounding_box(self) -> dict[str, float] | None:
        return self.box


def test_glide_defaults_to_the_center_and_takes_another_point(tmp_path: Path) -> None:
    rec, page, _ = make(tmp_path)
    box = {"x": 100.0, "y": 200.0, "width": 400.0, "height": 100.0}
    rec.glide(FakeLocator(box))  # type: ignore[arg-type]
    assert page.moves[-1] == pytest.approx((300.0, 250.0))
    rec.glide(FakeLocator(box), at=(0.9, 0.25))  # type: ignore[arg-type]
    assert page.moves[-1] == pytest.approx((460.0, 225.0))
    assert rec._pointer == pytest.approx((460.0, 225.0))


def test_glide_to_an_element_without_a_box_fails(tmp_path: Path) -> None:
    rec, _, _ = make(tmp_path)
    with pytest.raises(RuntimeError, match="no bounding box"):
        rec.glide(FakeLocator(None))  # type: ignore[arg-type]


def test_open_can_start_the_take_at_an_element_below_the_fold(tmp_path: Path) -> None:
    rec, page, _ = make(tmp_path)
    box = {"x": 0.0, "y": 900.0, "width": 10.0, "height": 10.0}
    rec.open("http://x/", start_at=FakeLocator(box))  # type: ignore[arg-type]
    assert page.evaluated[-1] == ("dy => window.scrollBy(0, dy)", 900 - harness.TOP_MARGIN_PX)


def test_scroll_to_wheels_the_element_to_the_top_margin(tmp_path: Path) -> None:
    rec, page, _ = make(tmp_path)
    rec.scroll_to(FakeLocator({"x": 0.0, "y": 500.0, "width": 10.0, "height": 10.0}))  # type: ignore[arg-type]
    assert sum(dy for _, dy in page.moves) == 500 - harness.TOP_MARGIN_PX
    with pytest.raises(RuntimeError, match="no bounding box"):
        rec.scroll_to(FakeLocator(None))  # type: ignore[arg-type]


class FakeField:
    def __init__(self, calls: list[tuple], name: str) -> None:
        self.calls = calls
        self.name = name

    def fill(self, value: str) -> None:
        self.calls.append(("fill", self.name, value))

    def press(self, key: str) -> None:
        self.calls.append(("press", self.name, key))


class FakeBanner:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def count(self) -> int:
        return 0 if self.text is None else 1

    def inner_text(self) -> str:
        assert self.text is not None
        return self.text


class SignInPage:
    """The login page: records what the harness does, and either lands on the
    home page or times out with `banner` showing."""

    def __init__(self, banner: str | None = None) -> None:
        self.calls: list[tuple] = []
        self.banner = banner

    def goto(self, url: str) -> None:
        self.calls.append(("goto", url))

    def get_by_label(self, label: str, exact: bool = False) -> FakeField:
        assert exact
        return FakeField(self.calls, label)

    def wait_for_url(self, url: str, timeout: float) -> None:
        self.calls.append(("wait_for_url", url, timeout))
        if self.banner is not None:
            raise TimeoutError("page.wait_for_url")

    def wait_for_load_state(self, state: str) -> None:
        self.calls.append(("wait_for_load_state", state))

    def locator(self, selector: str) -> FakeBanner:
        assert selector == "#login-error .error-banner"
        return FakeBanner(self.banner)


def test_sign_in_goes_through_the_login_page_and_waits_for_home() -> None:
    page = SignInPage()
    harness.sign_in(page, "http://x:3100", "me@example.com", "pw")  # type: ignore[arg-type]
    assert page.calls == [
        ("goto", "http://x:3100/login"),
        ("fill", "Email", "me@example.com"),
        ("fill", "Password", "pw"),
        ("press", "Password", "Enter"),
        ("wait_for_url", "http://x:3100/", harness.SIGN_IN_TIMEOUT_MS),
        ("wait_for_load_state", "networkidle"),
    ]


def test_a_refused_sign_in_says_why_instead_of_timing_out_on_the_address() -> None:
    page = SignInPage(banner="The email or password is incorrect.")
    with pytest.raises(RuntimeError, match="me@example.com failed: The email or password is incorrect"):
        harness.sign_in(page, "http://x:3100", "me@example.com", "pw")  # type: ignore[arg-type]
    assert ("wait_for_load_state", "networkidle") not in page.calls


def test_a_sign_in_with_no_banner_reports_no_answer() -> None:
    class Silent(SignInPage):
        def wait_for_url(self, url: str, timeout: float) -> None:
            raise TimeoutError("page.wait_for_url")

    with pytest.raises(RuntimeError, match="no answer from the API"):
        harness.sign_in(Silent(), "http://x", "a@b.c", "pw")  # type: ignore[arg-type]


def test_a_recording_cannot_start_without_credentials() -> None:
    parameters = inspect.signature(harness.recording).parameters
    assert parameters["email"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["password"].default is inspect.Parameter.empty
