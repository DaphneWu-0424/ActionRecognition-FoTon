from __future__ import annotations

import mimetypes
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class MediaInfo:
    kind: str
    mime_type: str
    width: int
    height: int
    duration_sec: float | None = None
    fps: float | None = None
    frame_count: int | None = None


def kind_from_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def probe_media(path: Path, expected_kind: str | None = None) -> MediaInfo:
    kind = expected_kind or kind_from_path(path)
    if kind == "image":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mime = Image.MIME.get(image.format or "")
        return MediaInfo(
            kind="image",
            mime_type=mime or mimetypes.guess_type(path.name)[0] or "image/jpeg",
            width=width,
            height=height,
        )
    if kind == "video":
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                return _probe_video_ffprobe(path, ffprobe)
            except (OSError, ValueError, subprocess.SubprocessError, KeyError):
                pass
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path.name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid video metadata: {path.name}")
            return MediaInfo(
                kind="video",
                mime_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                width=width,
                height=height,
                duration_sec=frame_count / fps,
                fps=fps,
                frame_count=frame_count,
            )
        finally:
            capture.release()
    raise ValueError(f"Unsupported media type: {path.name}")


def _probe_video_ffprobe(path: Path, executable: str) -> MediaInfo:
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    duration = float(payload["format"]["duration"])
    frame_count = int(stream.get("nb_frames") or round(duration * fps))
    if fps <= 0 or duration <= 0:
        raise ValueError("Invalid ffprobe metadata")
    return MediaInfo(
        kind="video",
        mime_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_sec=duration,
        fps=fps,
        frame_count=frame_count,
    )


def create_thumbnail(
    source: Path,
    kind: str,
    destination: Path,
    size: tuple[int, int] = (480, 270),
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "image":
        with Image.open(source) as image:
            frame = image.convert("RGB")
            frame.thumbnail(size)
            frame.save(destination, "JPEG", quality=85)
        return

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read thumbnail frame: {source.name}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(size)
        image.save(destination, "JPEG", quality=85)
    finally:
        capture.release()
