"""Tests for the recording harness's timing, with a fake page and clock.

Playwright is never imported (run: python -m pytest ../scripts/demos -q).
"""

from __future__ import annotations

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
