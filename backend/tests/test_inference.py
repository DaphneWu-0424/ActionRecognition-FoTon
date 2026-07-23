from __future__ import annotations

import pytest

from app.inference.base import LocalizationParams
from scripts.search_image_in_video import (
    build_window_starts,
    temporal_iou,
    temporal_nms,
)


def test_localization_params_accept_web_defaults() -> None:
    LocalizationParams().validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_fps", 0),
        ("window_sec", -1),
        ("stride_sec", 0),
        ("top_frame_ratio", 1.1),
        ("top_k", 2),
        ("nms_iou", -0.1),
    ],
)
def test_localization_params_reject_invalid_values(
    field: str,
    value: float,
) -> None:
    values = {
        "sample_fps": 2.0,
        "window_sec": 2.0,
        "stride_sec": 0.5,
        "top_frame_ratio": 0.5,
        "top_k": 1,
        "nms_iou": 0.3,
    }
    values[field] = value
    with pytest.raises(ValueError):
        LocalizationParams(**values).validate()


def test_window_starts_short_video() -> None:
    assert build_window_starts(1.0, 2.0, 0.5) == [0.0]


def test_window_starts_include_tail() -> None:
    starts = build_window_starts(5.1, 2.0, 1.0)
    assert starts[-1] == pytest.approx(3.1)


def test_temporal_iou() -> None:
    assert temporal_iou(
        {"start_sec": 0.0, "end_sec": 2.0},
        {"start_sec": 1.0, "end_sec": 3.0},
    ) == pytest.approx(1 / 3)


def test_temporal_nms_keeps_best_non_overlapping_window() -> None:
    windows = [
        {"start_sec": 0.0, "end_sec": 2.0, "score": 0.9},
        {"start_sec": 0.2, "end_sec": 2.2, "score": 0.8},
        {"start_sec": 4.0, "end_sec": 6.0, "score": 0.7},
    ]
    selected = temporal_nms(windows, top_k=2, iou_threshold=0.3)
    assert [item["score"] for item in selected] == [0.9, 0.7]
