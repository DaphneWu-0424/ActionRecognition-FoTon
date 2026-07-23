from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


ProgressCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]


class InferenceCancelled(RuntimeError):
    """Raised when a running localization job is cancelled."""


@dataclass(frozen=True)
class LocalizationParams:
    sample_fps: float = 2.0
    window_sec: float = 2.0
    stride_sec: float = 0.5
    top_frame_ratio: float = 0.5
    top_k: int = 1
    nms_iou: float = 0.3

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if self.stride_sec <= 0:
            raise ValueError("stride_sec must be positive")
        if not 0 < self.top_frame_ratio <= 1:
            raise ValueError("top_frame_ratio must be in (0, 1]")
        if self.top_k != 1:
            raise ValueError("The local web app only supports top_k=1")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou must be in [0, 1]")


@dataclass(frozen=True)
class VideoMetadata:
    native_fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    sampled_frames: int


@dataclass(frozen=True)
class LocalizationResult:
    start_sec: float
    end_sec: float
    score: float
    mean_score: float
    max_score: float
    best_frame_sec: float
    clip_path: Path
    best_frame_path: Path


@dataclass(frozen=True)
class LocalizationOutput:
    result: LocalizationResult
    video: VideoMetadata
    query_copy_path: Path
    frame_scores_path: Path
    window_scores_path: Path


class TemporalActionLocator(Protocol):
    def load(self) -> None: ...

    def localize(
        self,
        query_image: Path,
        target_video: Path,
        output_dir: Path,
        params: LocalizationParams,
        on_progress: ProgressCallback = ...,
        is_cancelled: CancelCheck = ...,
    ) -> LocalizationOutput: ...


def noop_progress(stage: str, progress: float) -> None:
    del stage, progress


def never_cancelled() -> bool:
    return False
