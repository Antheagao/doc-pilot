"""Encode a recording into the demo deliverables.

Pure ffmpeg plumbing, no Playwright import, so the filter-graph and sidecar
math is unit-testable without a browser or ffmpeg. The harness
(harness.py) writes the recording as a list of timed JPEG frames (an ffmpeg
concat file, see frame_list) plus a JSON sidecar; encode_all() turns the
pair into <slug>.gif, <slug>.webm, <slug>.mp4 and <slug>.jpg:

- the lead-in (blank page before the first meaningful paint) is trimmed;
- each fast-forward segment is sped up with setpts and the pieces are
  concatenated;
- the GIF is a two-pass palette encode, retried smaller until it fits.

Times in the sidecar are seconds from the first frame.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIF_WIDTH = 960
MAX_GIF_SECONDS = 15.0
MAX_GIF_BYTES = 1_000_000

# (fps, palette colors) tried in order until the GIF is under MAX_GIF_BYTES.
# The width stays at GIF_WIDTH on every rung, because `demo.py check` holds
# committed GIFs to it; frame rate and palette size are the levers instead.
GIF_LADDER = ((12, 256), (10, 128), (8, 64))
VIDEO_FPS = 30
# The frame list holds a frame only when the page changed; the timeline
# graph first resamples it to this constant rate (a multiple of the GIF and
# video rates), so every trim point has frames on both sides and a held
# last frame lasts to the end of the recording.
SOURCE_FPS = 60
POSTER_SECONDS = 1.0

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class EncodeError(Exception):
    """A failure to report on stderr and turn into a non-zero exit code."""


@dataclass(frozen=True)
class Segment:
    """A stretch of the recording played `factor` times faster."""

    start: float
    end: float
    factor: float


@dataclass(frozen=True)
class Piece:
    """One contiguous slice of the recording in the final timeline."""

    start: float
    end: float
    factor: float = 1.0


@dataclass(frozen=True)
class Sidecar:
    """What the harness knows about a recording that the frames themselves
    do not say: where the content starts and which stretches to speed up."""

    slug: str
    lead_in: float
    duration: float
    segments: tuple[Segment, ...] = ()

    def validate(self) -> None:
        if self.lead_in < 0 or self.duration <= self.lead_in:
            raise EncodeError(
                f"{self.slug}: lead-in {self.lead_in} must be in [0, duration {self.duration})"
            )
        previous_end = self.lead_in
        for segment in self.segments:
            if segment.factor <= 0:
                raise EncodeError(f"{self.slug}: speed factor must be positive")
            if segment.end <= segment.start:
                raise EncodeError(f"{self.slug}: empty speed segment")
            if segment.start < previous_end:
                raise EncodeError(
                    f"{self.slug}: speed segments must be ordered, not overlap, "
                    "and start after the lead-in"
                )
            previous_end = segment.end

    def to_json(self) -> str:
        return json.dumps(
            {
                "slug": self.slug,
                "lead_in": self.lead_in,
                "duration": self.duration,
                "segments": [
                    {"start": s.start, "end": s.end, "factor": s.factor}
                    for s in self.segments
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> Sidecar:
        data = json.loads(text)
        sidecar = cls(
            slug=data["slug"],
            lead_in=float(data["lead_in"]),
            duration=float(data["duration"]),
            segments=tuple(
                Segment(float(s["start"]), float(s["end"]), float(s["factor"]))
                for s in data["segments"]
            ),
        )
        sidecar.validate()
        return sidecar


def pieces(sidecar: Sidecar) -> list[Piece]:
    """The slices of the recording, in order, that make the final timeline:
    everything from the lead-in to the end, with the fast-forward segments
    carrying their speed factor. A segment running past the end of the
    recording is clipped to it."""
    sidecar.validate()
    result = []
    cursor = sidecar.lead_in
    for segment in sidecar.segments:
        end = min(segment.end, sidecar.duration)
        if end <= segment.start:
            continue
        if segment.start > cursor:
            result.append(Piece(cursor, segment.start))
        result.append(Piece(segment.start, end, segment.factor))
        cursor = end
    if cursor < sidecar.duration:
        result.append(Piece(cursor, sidecar.duration))
    return result


def output_duration(slices: list[Piece]) -> float:
    """Length in seconds of the timeline the slices make."""
    return sum((piece.end - piece.start) / piece.factor for piece in slices)


def timeline_graph(slices: list[Piece], out_label: str = "tl") -> str:
    """The filter graph that trims, speeds up and concatenates the slices of
    input 0 into the labelled output pad."""
    if not slices:
        raise EncodeError("nothing to encode: the timeline is empty")
    stages = [
        f"trim=start={piece.start:.3f}:end={piece.end:.3f},"
        f"setpts=(PTS-STARTPTS)/{piece.factor:g}"
        for piece in slices
    ]
    if len(stages) == 1:
        return f"[0:v]fps={SOURCE_FPS},{stages[0]}[{out_label}]"
    split_labels = "".join(f"[s{i}]" for i in range(len(stages)))
    parts = [f"[0:v]fps={SOURCE_FPS},split={len(stages)}{split_labels}"]
    parts += [f"[s{i}]{stage}[p{i}]" for i, stage in enumerate(stages)]
    joined = "".join(f"[p{i}]" for i in range(len(stages)))
    parts.append(f"{joined}concat=n={len(stages)}:v=1:a=0[{out_label}]")
    return ";".join(parts)


def gif_scale(fps: int) -> str:
    return f"fps={fps},scale={GIF_WIDTH}:-1:flags=lanczos"


def palette_command(
    frames: Path, slices: list[Piece], palette: Path, fps: int, colors: int
) -> list[str]:
    """Pass 1 of the GIF: a palette built from the frames that change."""
    graph = (
        f"{timeline_graph(slices)};"
        f"[tl]{gif_scale(fps)},palettegen=max_colors={colors}:stats_mode=diff[pal]"
    )
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source_args(frames),
        "-filter_complex",
        graph,
        "-map",
        "[pal]",
        "-update",
        "1",
        "-frames:v",
        "1",
        str(palette),
    ]


def gif_command(
    frames: Path, slices: list[Piece], palette: Path, gif: Path, fps: int
) -> list[str]:
    """Pass 2 of the GIF: map the frames onto the palette."""
    graph = (
        f"{timeline_graph(slices)};"
        f"[tl]{gif_scale(fps)}[x];"
        "[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle[out]"
    )
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source_args(frames),
        "-i",
        str(palette),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-loop",
        "0",
        str(gif),
    ]


def webm_command(frames: Path, slices: list[Piece], webm: Path) -> list[str]:
    graph = f"{timeline_graph(slices)};[tl]fps={VIDEO_FPS}[out]"
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source_args(frames),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "36",
        "-b:v",
        "0",
        "-an",
        str(webm),
    ]


def mp4_command(frames: Path, slices: list[Piece], mp4: Path) -> list[str]:
    graph = f"{timeline_graph(slices)};[tl]fps={VIDEO_FPS}[out]"
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source_args(frames),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(mp4),
    ]


def poster_command(
    frames: Path, slices: list[Piece], jpg: Path, at_seconds: float
) -> list[str]:
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source_args(frames),
        "-filter_complex",
        timeline_graph(slices),
        "-map",
        "[tl]",
        "-ss",
        f"{at_seconds:.3f}",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(jpg),
    ]


def frame_list(frames: list[tuple[float, str]], end: float) -> str:
    """The ffmpeg concat-demuxer list that replays timed frames.

    frames is (seconds from the first frame, file name) in order; each frame
    stays on screen until the next, the last until `end`. The final file is
    listed again without a duration, because the concat demuxer ignores the
    duration of the last entry. Every entry declares a 1000 fps source: the
    image demuxer otherwise opens each JPEG at 25 fps, and the concat
    demuxer rounds every frame time to that 40 ms grid and drops frames
    that land on the same tick."""
    if not frames:
        raise EncodeError("the recording has no frames")
    lines = []
    for index, (start, name) in enumerate(frames):
        following = frames[index + 1][0] if index + 1 < len(frames) else end
        lines.append(f"file '{name}'")
        lines.append("option framerate 1000")
        lines.append(f"duration {max(following - start, 0.001):.3f}")
    lines.append(f"file '{frames[-1][1]}'")
    lines.append("option framerate 1000")
    return "\n".join(lines) + "\n"


def source_args(frames: Path) -> list[str]:
    """ffmpeg input arguments that read a frame list written by frame_list."""
    return ["-f", "concat", "-safe", "0", "-i", str(frames)]


def require_tools() -> None:
    """Fail fast, with the fix, when ffmpeg or ffprobe is not on PATH."""
    missing = [tool for tool in (FFMPEG, FFPROBE) if shutil.which(tool) is None]
    if missing:
        raise EncodeError(
            f"{' and '.join(missing)} not found on PATH; install ffmpeg "
            "(winget install Gyan.FFmpeg) and open a new terminal"
        )


def run_tool(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-5:]
        raise EncodeError(f"{cmd[0]} failed: " + " | ".join(tail))


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    size: int


def probe(path: Path) -> MediaInfo:
    """Duration, width and file size of a media file, via ffprobe."""
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise EncodeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return MediaInfo(
        duration=float(data["format"]["duration"]),
        width=int(data["streams"][0]["width"]),
        size=path.stat().st_size,
    )


def gif_problems(info: MediaInfo) -> list[str]:
    """Reasons a committed GIF breaks the limits; empty when it is within them."""
    problems = []
    if info.duration > MAX_GIF_SECONDS:
        problems.append(f"duration {info.duration:.1f}s is over {MAX_GIF_SECONDS:g}s")
    if info.size >= MAX_GIF_BYTES:
        problems.append(f"size {info.size} bytes is not under {MAX_GIF_BYTES}")
    if info.width != GIF_WIDTH:
        problems.append(f"width {info.width}px is not {GIF_WIDTH}px")
    return problems


def encode_gif(
    frames: Path, slices: list[Piece], gif: Path, work_dir: Path
) -> tuple[int, int]:
    """Write the GIF, retrying down GIF_LADDER until it is under
    MAX_GIF_BYTES; returns the (fps, colors) that fit."""
    palette = work_dir / f"{gif.stem}.palette.png"
    try:
        for fps, colors in GIF_LADDER:
            run_tool(palette_command(frames, slices, palette, fps, colors))
            run_tool(gif_command(frames, slices, palette, gif, fps))
            if gif.stat().st_size < MAX_GIF_BYTES:
                return fps, colors
    finally:
        palette.unlink(missing_ok=True)
    raise EncodeError(
        f"{gif.name} is still {gif.stat().st_size} bytes at {fps} fps and {colors} colors; "
        "shorten the scenario or speed up more of it"
    )


def encode_all(frames: Path, sidecar_path: Path, out_dir: Path) -> dict[str, Path]:
    """Encode the recording (a frame list, see frame_list) into <slug>.gif/.webm/.mp4/.jpg in out_dir."""
    require_tools()
    sidecar = Sidecar.from_json(sidecar_path.read_text(encoding="utf-8"))
    slices = pieces(sidecar)
    duration = output_duration(slices)
    if duration > MAX_GIF_SECONDS:
        raise EncodeError(
            f"{sidecar.slug} plays for {duration:.1f}s, over the "
            f"{MAX_GIF_SECONDS:g}s limit; shorten the scenario or speed up more of it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        kind: out_dir / f"{sidecar.slug}.{kind}"
        for kind in ("gif", "webm", "mp4", "jpg")
    }
    encode_gif(frames, slices, paths["gif"], out_dir)
    run_tool(webm_command(frames, slices, paths["webm"]))
    run_tool(mp4_command(frames, slices, paths["mp4"]))
    poster_at = min(POSTER_SECONDS, duration / 2)
    run_tool(poster_command(frames, slices, paths["jpg"], poster_at))
    return paths
