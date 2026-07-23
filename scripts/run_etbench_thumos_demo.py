from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-video image-to-untrimmed-video localization."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ActionRecognition"),
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Optional action keyword, such as BaseballPitch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)

    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "samples", "annotations", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        f"Unsupported annotation JSON structure: {path}"
    )


def get_video_value(item: dict[str, Any]) -> str:
    for key in ("video", "video_path", "filename", "file"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise KeyError(f"No video path field in item: {item}")


def get_query_text(item: dict[str, Any]) -> str:
    for key in (
        "q",
        "query",
        "question",
        "prompt",
        "action",
        "label",
        "class_name",
        "category",
    ):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return "unknown_action"


def normalize_intervals(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    intervals: list[tuple[float, float]] = []

    if isinstance(value, dict):
        for key in ("intervals", "segments", "timestamps", "tgt"):
            if key in value:
                return normalize_intervals(value[key])

        start = value.get("start", value.get("start_sec"))
        end = value.get("end", value.get("end_sec"))

        if start is not None and end is not None:
            intervals.append((float(start), float(end)))

        return intervals

    if not isinstance(value, list):
        return intervals

    if (
        len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        start, end = float(value[0]), float(value[1])

        if end > start:
            intervals.append((start, end))

        return intervals

    for entry in value:
        intervals.extend(normalize_intervals(entry))

    return intervals


def get_intervals(item: dict[str, Any]) -> list[tuple[float, float]]:
    for key in (
        "tgt",
        "target",
        "timestamps",
        "segments",
        "intervals",
        "ground_truth",
    ):
        if key in item:
            intervals = normalize_intervals(item[key])

            if intervals:
                return intervals

    return []


def canonical_group_key(item: dict[str, Any]) -> str:
    for key in (
        "action",
        "label",
        "class_name",
        "category",
        "class",
    ):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    query = get_query_text(item).lower()
    query = re.sub(r"\s+", " ", query).strip()

    return query


def resolve_video_path(
    video_value: str,
    videos_dir: Path,
) -> Path:
    raw = Path(video_value)

    candidates = [
        videos_dir / raw.name,
        videos_dir / raw,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    matches = list(videos_dir.rglob(raw.name))

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Cannot resolve video '{video_value}' under {videos_dir}"
    )


def extract_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp_sec * 1000.0,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                f"Cannot read {video_path} at {timestamp_sec:.3f}s"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Cannot save image: {output_path}")

    finally:
        cap.release()


def temporal_iou(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    intersection = max(
        0.0,
        min(first[1], second[1]) - max(first[0], second[0]),
    )

    union = (
        first[1] - first[0]
        + second[1] - second[0]
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def best_gt_iou(
    prediction: tuple[float, float],
    ground_truth: list[tuple[float, float]],
) -> float:
    return max(
        (temporal_iou(prediction, gt) for gt in ground_truth),
        default=0.0,
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    annotation_path = (
        root
        / "data"
        / "etbench_thumos14"
        / "annotations"
        / "txt"
        / "tal_thumos14.json"
    )

    videos_dir = (
        root
        / "data"
        / "etbench_thumos14"
        / "videos"
        / "thumos14"
    )

    search_script = (
        root
        / "scripts"
        / "search_image_in_video.py"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    if not videos_dir.exists():
        raise FileNotFoundError(videos_dir)

    if not search_script.exists():
        raise FileNotFoundError(search_script)

    raw_items = load_json(annotation_path)

    valid_items: list[dict[str, Any]] = []

    for raw_item in raw_items:
        try:
            video_path = resolve_video_path(
                get_video_value(raw_item),
                videos_dir,
            )
        except (KeyError, FileNotFoundError):
            continue

        intervals = get_intervals(raw_item)

        if not intervals:
            continue

        valid_items.append(
            {
                "raw": raw_item,
                "video_path": video_path,
                "intervals": intervals,
                "query_text": get_query_text(raw_item),
                "group_key": canonical_group_key(raw_item),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in valid_items:
        groups[item["group_key"]].append(item)

    candidate_groups = {
        key: items
        for key, items in groups.items()
        if len({item["video_path"] for item in items}) >= 2
    }

    if args.keyword:
        keyword = args.keyword.lower()

        candidate_groups = {
            key: items
            for key, items in candidate_groups.items()
            if keyword in key
            or any(
                keyword in item["query_text"].lower()
                for item in items
            )
        }

    if not candidate_groups:
        print("Available groups with at least two videos:")

        for key, items in sorted(
            groups.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:30]:
            print(f"{len(items):3d}  {key[:120]}")

        raise RuntimeError(
            "No action group with at least two distinct videos was found."
        )

    rng = random.Random(args.seed)

    group_key = rng.choice(sorted(candidate_groups))
    group_items = candidate_groups[group_key]

    unique_by_video: dict[Path, dict[str, Any]] = {}

    for item in group_items:
        unique_by_video.setdefault(item["video_path"], item)

    source_item, target_item = rng.sample(
        list(unique_by_video.values()),
        k=2,
    )

    source_interval = source_item["intervals"][0]
    query_timestamp = (
        source_interval[0] + source_interval[1]
    ) / 2.0

    experiment_name = (
        re.sub(r"[^a-zA-Z0-9_-]+", "_", group_key)[:50]
        or "action"
    )

    output_dir = (
        root
        / "outputs"
        / f"etbench_thumos_{experiment_name}_seed_{args.seed}"
    )

    search_output_dir = output_dir / "search"
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = output_dir / "query.png"

    extract_frame(
        video_path=source_item["video_path"],
        timestamp_sec=query_timestamp,
        output_path=query_path,
    )

    manifest = {
        "group_key": group_key,
        "query_text": source_item["query_text"],
        "query_source_video": str(source_item["video_path"]),
        "query_source_interval": list(source_interval),
        "query_timestamp_sec": query_timestamp,
        "query_image": str(query_path.resolve()),
        "target_video": str(target_item["video_path"]),
        "target_ground_truth": [
            list(interval)
            for interval in target_item["intervals"]
        ],
    }

    (output_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(search_script),
        "--root",
        str(root),
        "--query-image",
        str(query_path),
        "--video",
        str(target_item["video_path"]),
        "--output-dir",
        str(search_output_dir),
        "--sample-fps",
        str(args.sample_fps),
        "--window-sec",
        str(args.window_sec),
        "--stride-sec",
        str(args.stride_sec),
        "--top-k",
        str(args.top_k),
        "--threads",
        str(args.threads),
    ]

    print("=" * 72)
    print("Selected cross-video localization pair")
    print("=" * 72)
    print("Action group:", group_key)
    print("Query text:", source_item["query_text"])
    print("Query source:", source_item["video_path"])
    print("Query source interval:", source_interval)
    print("Query timestamp:", query_timestamp)
    print("Target video:", target_item["video_path"])
    print("Target GT:", target_item["intervals"])
    print()
    print("Running temporal search...")

    subprocess.run(command, check=True)

    prediction_path = search_output_dir / "predictions.json"

    prediction_data = json.loads(
        prediction_path.read_text(encoding="utf-8")
    )

    predictions = prediction_data.get("predictions", [])
    ground_truth = target_item["intervals"]

    evaluated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        interval = (
            float(prediction["start_sec"]),
            float(prediction["end_sec"]),
        )

        evaluated_predictions.append(
            {
                **prediction,
                "best_gt_iou": best_gt_iou(
                    interval,
                    ground_truth,
                ),
            }
        )

    top1_iou = (
        evaluated_predictions[0]["best_gt_iou"]
        if evaluated_predictions
        else 0.0
    )

    topk_best_iou = max(
        (
            item["best_gt_iou"]
            for item in evaluated_predictions
        ),
        default=0.0,
    )

    evaluation = {
        "action_group": group_key,
        "query_text": source_item["query_text"],
        "query_image": str(query_path.resolve()),
        "target_video": str(target_item["video_path"]),
        "ground_truth_intervals": [
            list(interval)
            for interval in ground_truth
        ],
        "top1_iou": top1_iou,
        "topk_best_iou": topk_best_iou,
        "top1_hit_iou_0.3": top1_iou >= 0.3,
        "top1_hit_iou_0.5": top1_iou >= 0.5,
        "topk_hit_iou_0.3": topk_best_iou >= 0.3,
        "topk_hit_iou_0.5": topk_best_iou >= 0.5,
        "predictions": evaluated_predictions,
    }

    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Evaluation")
    print("=" * 72)
    print("Top-1 IoU:       ", f"{top1_iou:.4f}")
    print("Top-K best IoU:  ", f"{topk_best_iou:.4f}")
    print("Top-1 hit@0.3:   ", top1_iou >= 0.3)
    print("Top-1 hit@0.5:   ", top1_iou >= 0.5)
    print("Top-K hit@0.3:   ", topk_best_iou >= 0.3)
    print("Top-K hit@0.5:   ", topk_best_iou >= 0.5)
    print("Output:          ", output_dir)


if __name__ == "__main__":
    main()