"""Tests for the encoder's filter graph, sidecar math and size ladder.

No ffmpeg or Playwright is needed (run: python -m pytest ../scripts/demos -q).
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import encode
import pytest
from encode import Piece, Segment, Sidecar


def sidecar(
    *segments: Segment, lead_in: float = 1.0, duration: float = 10.0
) -> Sidecar:
    return Sidecar("demo", lead_in, duration, tuple(segments))


def test_no_segments_is_one_piece_from_the_lead_in_to_the_end() -> None:
    assert encode.pieces(sidecar()) == [Piece(1.0, 10.0)]


def test_segments_split_the_timeline_and_carry_their_factor() -> None:
    result = encode.pieces(sidecar(Segment(3.0, 5.0, 4), Segment(7.0, 9.0, 2)))
    assert result == [
        Piece(1.0, 3.0),
        Piece(3.0, 5.0, 4),
        Piece(5.0, 7.0),
        Piece(7.0, 9.0, 2),
        Piece(9.0, 10.0),
    ]


def test_a_segment_at_the_start_or_end_leaves_no_empty_piece() -> None:
    result = encode.pieces(sidecar(Segment(1.0, 4.0, 4), Segment(8.0, 10.0, 4)))
    assert result == [Piece(1.0, 4.0, 4), Piece(4.0, 8.0), Piece(8.0, 10.0, 4)]


def test_a_segment_past_the_end_is_clipped() -> None:
    result = encode.pieces(sidecar(Segment(8.0, 12.0, 4)))
    assert result == [Piece(1.0, 8.0), Piece(8.0, 10.0, 4)]


def test_output_duration_divides_each_piece_by_its_factor() -> None:
    slices = [Piece(1.0, 3.0), Piece(3.0, 7.0, 4), Piece(7.0, 10.0)]
    assert encode.output_duration(slices) == pytest.approx(2 + 1 + 3)


def test_pieces_cover_the_recording_without_gaps_or_overlap() -> None:
    result = encode.pieces(sidecar(Segment(2.0, 3.0, 4), Segment(3.0, 6.0, 8)))
    assert result[0].start == 1.0
    assert result[-1].end == 10.0
    for before, after in itertools.pairwise(result):
        assert before.end == after.start


@pytest.mark.parametrize(
    "bad",
    [
        sidecar(lead_in=10.0),  # lead-in swallows the recording
        sidecar(lead_in=-1.0),
        sidecar(Segment(0.5, 2.0, 4)),  # starts before the lead-in
        sidecar(Segment(3.0, 5.0, 4), Segment(4.0, 6.0, 4)),  # overlap
        sidecar(Segment(5.0, 6.0, 4), Segment(2.0, 3.0, 4)),  # out of order
        sidecar(Segment(3.0, 3.0, 4)),  # empty
        sidecar(Segment(3.0, 4.0, 0)),  # no speed
    ],
)
def test_invalid_sidecars_are_refused(bad: Sidecar) -> None:
    with pytest.raises(encode.EncodeError):
        bad.validate()


def test_sidecar_round_trips_through_json() -> None:
    original = sidecar(Segment(3.0, 5.0, 4))
    assert Sidecar.from_json(original.to_json()) == original


def test_single_piece_graph_trims_and_resets_pts() -> None:
    graph = encode.timeline_graph([Piece(1.0, 9.5)])
    assert graph == (
        "[0:v]fps=60,trim=start=1.000:end=9.500,setpts=(PTS-STARTPTS)/1[tl]"
    )


def test_multi_piece_graph_splits_speeds_up_and_concatenates() -> None:
    graph = encode.timeline_graph([Piece(1.0, 3.0), Piece(3.0, 7.0, 4), Piece(7, 9)])
    assert graph.startswith("[0:v]fps=60,split=3[s0][s1][s2];")
    assert "[s1]trim=start=3.000:end=7.000,setpts=(PTS-STARTPTS)/4[p1]" in graph
    assert graph.endswith("[p0][p1][p2]concat=n=3:v=1:a=0[tl]")


def test_empty_timeline_is_refused() -> None:
    with pytest.raises(encode.EncodeError):
        encode.timeline_graph([])


def test_frame_list_holds_each_frame_until_the_next_and_the_last_to_the_end() -> None:
    text = encode.frame_list([(0.0, "a.jpg"), (0.5, "b.jpg"), (2.0, "c.jpg")], 3.0)
    assert text == (
        "file 'a.jpg'\nduration 0.500\n"
        "file 'b.jpg'\nduration 1.500\n"
        "file 'c.jpg'\nduration 1.000\n"
        "file 'c.jpg'\n"
    )


def test_frame_list_needs_frames() -> None:
    with pytest.raises(encode.EncodeError):
        encode.frame_list([], 1.0)


SLICES = [Piece(1.0, 9.0)]


def test_palette_pass_uses_diff_stats() -> None:
    cmd = encode.palette_command(Path("f.txt"), SLICES, Path("p.png"), 12, 960)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "fps=12,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff" in graph
    assert cmd[-1] == "p.png"


def test_gif_pass_uses_the_palette_with_bayer_dither() -> None:
    cmd = encode.gif_command(
        Path("f.txt"), SLICES, Path("p.png"), Path("o.gif"), 10, 840
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "fps=10,scale=840:-1:flags=lanczos" in graph
    assert "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" in graph
    assert cmd.count("-i") == 2
    assert cmd[-1] == "o.gif"


def test_webm_and_mp4_settings() -> None:
    webm = encode.webm_command(Path("f.txt"), SLICES, Path("o.webm"))
    mp4 = encode.mp4_command(Path("f.txt"), SLICES, Path("o.mp4"))
    assert webm[-8:-1] == ["-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-an"]
    assert mp4[mp4.index("-crf") :][:4] == ["-crf", "26", "-pix_fmt", "yuv420p"]
    assert "+faststart" in mp4 and "-an" in mp4


def test_commands_read_the_frame_list_with_the_concat_demuxer() -> None:
    cmd = encode.poster_command(Path("f.txt"), SLICES, Path("o.jpg"), 1.0)
    start = cmd.index("-f")
    assert cmd[start : start + 6] == ["-f", "concat", "-safe", "0", "-i", "f.txt"]


@pytest.mark.parametrize(
    ("duration", "size", "width", "ok"),
    [
        (15.0, 999_999, 960, True),
        (15.01, 500_000, 960, False),
        (10.0, 1_000_000, 960, False),
        (10.0, 500_000, 840, False),
    ],
)
def test_gif_limits(duration: float, size: int, width: int, ok: bool) -> None:
    problems = encode.gif_problems(encode.MediaInfo(duration, width, size))
    assert (problems == []) is ok


def fake_encoder(
    monkeypatch: pytest.MonkeyPatch, sizes: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    """Replace ffmpeg: a GIF pass writes `sizes[(fps, width)]` bytes."""
    attempts: list[tuple[int, int]] = []

    def run_tool(cmd: list[str]) -> None:
        graph = cmd[cmd.index("-filter_complex") + 1]
        fps, width = map(int, re.search(r"fps=(\d+),scale=(\d+):", graph).groups())
        if cmd[-1].endswith(".gif"):
            attempts.append((fps, width))
            Path(cmd[-1]).write_bytes(b"x" * sizes[(fps, width)])
        else:
            Path(cmd[-1]).write_bytes(b"palette")

    monkeypatch.setattr(encode, "run_tool", run_tool)
    return attempts


def test_gif_ladder_stops_at_the_first_size_that_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = fake_encoder(
        monkeypatch, {(12, 960): 1_400_000, (10, 960): 900_000, (10, 840): 1}
    )
    chosen = encode.encode_gif(Path("f.txt"), SLICES, tmp_path / "a.gif", tmp_path)
    assert chosen == (10, 960)
    assert attempts == [(12, 960), (10, 960)]
    assert list(tmp_path.glob("*.palette.png")) == []


def test_gif_ladder_falls_to_the_smallest_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_encoder(
        monkeypatch, {(12, 960): 2_000_000, (10, 960): 1_500_000, (10, 840): 999_000}
    )
    chosen = encode.encode_gif(Path("f.txt"), SLICES, tmp_path / "a.gif", tmp_path)
    assert chosen == (10, 840)


def test_gif_ladder_fails_when_nothing_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_encoder(
        monkeypatch, {(12, 960): 3_000_000, (10, 960): 2_000_000, (10, 840): 1_200_000}
    )
    with pytest.raises(encode.EncodeError, match="still 1200000 bytes"):
        encode.encode_gif(Path("f.txt"), SLICES, tmp_path / "a.gif", tmp_path)


def test_missing_ffmpeg_fails_fast_with_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(encode.shutil, "which", lambda tool: None)
    with pytest.raises(encode.EncodeError, match="ffmpeg and ffprobe not found"):
        encode.require_tools()
