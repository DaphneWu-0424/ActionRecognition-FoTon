from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch cross-video image-to-untrimmed-video "
            "localization on E.T. Bench THUMOS14."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ActionRecognition"),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="只测试包含该关键词的动作，例如 BaseballPitch。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--stride-sec",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--full-video-mode",
        choices=["hardlink", "copy", "reference"],
        default="hardlink",
        help=(
            "hardlink: 优先建立 NTFS 硬链接；"
            "copy: 复制完整视频；"
            "reference: 只在 JSON 中记录原路径。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已经存在 evaluation.json 的 episode。",
    )

    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "data",
            "samples",
            "annotations",
            "items",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        f"Unsupported annotation structure: {path}"
    )


def get_video_value(item: dict[str, Any]) -> str:
    for key in (
        "video",
        "video_path",
        "filename",
        "file",
    ):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise KeyError("No video path field.")


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


def normalize_intervals(
    value: Any,
) -> list[tuple[float, float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    if isinstance(value, dict):
        for key in (
            "intervals",
            "segments",
            "timestamps",
            "tgt",
        ):
            if key in value:
                return normalize_intervals(value[key])

        start = value.get(
            "start",
            value.get("start_sec"),
        )

        end = value.get(
            "end",
            value.get("end_sec"),
        )

        if start is not None and end is not None:
            start = float(start)
            end = float(end)

            if end > start:
                return [(start, end)]

        return []

    if not isinstance(value, list):
        return []

    if (
        len(value) == 2
        and all(
            isinstance(item, (int, float))
            for item in value
        )
    ):
        start = float(value[0])
        end = float(value[1])

        return (
            [(start, end)]
            if end > start
            else []
        )

    output: list[tuple[float, float]] = []

    for entry in value:
        output.extend(
            normalize_intervals(entry)
        )

    return output


def get_intervals(
    item: dict[str, Any],
) -> list[tuple[float, float]]:
    for key in (
        "tgt",
        "target",
        "timestamps",
        "segments",
        "intervals",
        "ground_truth",
    ):
        if key not in item:
            continue

        intervals = normalize_intervals(
            item[key]
        )

        if intervals:
            return intervals

    return []


def canonical_group_key(
    item: dict[str, Any],
) -> str:
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
    query = re.sub(r"\s+", " ", query)

    return query.strip()


def safe_name(text: str) -> str:
    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        text,
    )

    value = value.strip("_")

    return value[:60] or "action"


def resolve_video_path(
    video_value: str,
    videos_dir: Path,
) -> Path:
    raw = Path(
        video_value.replace("\\", "/")
    )

    candidates = [
        videos_dir / raw.name,
        videos_dir / raw,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = list(
        videos_dir.rglob(raw.name)
    )

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Cannot resolve video: {video_value}"
    )


def extract_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp_sec * 1000.0,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                f"Cannot read {video_path} "
                f"at {timestamp_sec:.3f}s"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(output_path),
            frame,
        ):
            raise RuntimeError(
                f"Cannot save query image: "
                f"{output_path}"
            )

    finally:
        cap.release()


def materialize_full_video(
    source: Path,
    destination: Path,
    mode: str,
) -> dict[str, str]:
    source = source.resolve()

    if mode == "reference":
        return {
            "requested_mode": mode,
            "actual_mode": "reference",
            "source": str(source),
            "output": str(source),
        }

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    if mode == "hardlink":
        try:
            os.link(
                source,
                destination,
            )

            return {
                "requested_mode": mode,
                "actual_mode": "hardlink",
                "source": str(source),
                "output": str(
                    destination.resolve()
                ),
            }

        except OSError as exc:
            print(
                "[WARNING] Hardlink failed; "
                f"falling back to copy: {exc}"
            )

    shutil.copy2(
        source,
        destination,
    )

    return {
        "requested_mode": mode,
        "actual_mode": "copy",
        "source": str(source),
        "output": str(
            destination.resolve()
        ),
    }


def temporal_iou(
    prediction: tuple[float, float],
    target: tuple[float, float],
) -> float:
    intersection = max(
        0.0,
        min(prediction[1], target[1])
        - max(prediction[0], target[0]),
    )

    union = (
        prediction[1] - prediction[0]
        + target[1] - target[0]
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
        (
            temporal_iou(
                prediction,
                interval,
            )
            for interval in ground_truth
        ),
        default=0.0,
    )


def build_candidate_pairs(
    items: list[dict[str, Any]],
    keyword: str | None,
) -> list[
    tuple[
        str,
        dict[str, Any],
        dict[str, Any],
    ]
]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in items:
        groups[
            item["group_key"]
        ].append(item)

    pairs: list[
        tuple[
            str,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    keyword_lower = (
        keyword.lower()
        if keyword
        else None
    )

    for group_key, group_items in groups.items():
        if (
            keyword_lower
            and keyword_lower not in group_key.lower()
            and not any(
                keyword_lower
                in item["query_text"].lower()
                for item in group_items
            )
        ):
            continue

        unique_by_video: dict[
            Path,
            dict[str, Any],
        ] = {}

        for item in group_items:
            unique_by_video.setdefault(
                item["video_path"],
                item,
            )

        unique_items = list(
            unique_by_video.values()
        )

        if len(unique_items) < 2:
            continue

        for source_index, source_item in enumerate(
            unique_items
        ):
            for target_index, target_item in enumerate(
                unique_items
            ):
                if source_index == target_index:
                    continue

                pairs.append(
                    (
                        group_key,
                        source_item,
                        target_item,
                    )
                )

    return pairs


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be positive."
        )

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

    if args.output_dir is None:
        output_root = (
            root
            / "outputs"
            / (
                "etbench_thumos_batch"
                f"_seed_{args.seed}"
            )
        )
    else:
        output_root = (
            args.output_dir.resolve()
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not annotation_path.exists():
        raise FileNotFoundError(
            annotation_path
        )

    if not videos_dir.exists():
        raise FileNotFoundError(
            videos_dir
        )

    if not search_script.exists():
        raise FileNotFoundError(
            search_script
        )

    raw_items = load_json(
        annotation_path
    )

    valid_items: list[
        dict[str, Any]
    ] = []

    for raw_item in raw_items:
        try:
            video_path = resolve_video_path(
                get_video_value(raw_item),
                videos_dir,
            )

            intervals = get_intervals(
                raw_item
            )

            if not intervals:
                continue

            valid_items.append(
                {
                    "raw": raw_item,
                    "video_path": video_path,
                    "intervals": intervals,
                    "query_text": get_query_text(
                        raw_item
                    ),
                    "group_key": canonical_group_key(
                        raw_item
                    ),
                }
            )

        except (
            KeyError,
            FileNotFoundError,
            ValueError,
        ):
            continue

    pairs = build_candidate_pairs(
        items=valid_items,
        keyword=args.keyword,
    )

    if not pairs:
        raise RuntimeError(
            "No valid cross-video pairs found."
        )

    rng = random.Random(
        args.seed
    )

    rng.shuffle(pairs)

    if args.episodes > len(pairs):
        print(
            "[WARNING] Requested episodes exceed "
            "available ordered pairs. Pairs will repeat."
        )

        original_pairs = list(pairs)

        while len(pairs) < args.episodes:
            extension = list(
                original_pairs
            )

            rng.shuffle(extension)
            pairs.extend(extension)

    selected_pairs = pairs[
        : args.episodes
    ]

    summary_rows: list[
        dict[str, Any]
    ] = []

    completed = 0
    failures = 0

    for episode_index, (
        group_key,
        source_item,
        target_item,
    ) in enumerate(selected_pairs):
        episode_name = (
            f"episode_{episode_index:03d}_"
            f"{safe_name(group_key)}"
        )

        episode_dir = (
            output_root
            / episode_name
        )

        evaluation_path = (
            episode_dir
            / "evaluation.json"
        )

        if (
            args.resume
            and evaluation_path.exists()
        ):
            print(
                f"[SKIP] {episode_name}"
            )

            evaluation = json.loads(
                evaluation_path.read_text(
                    encoding="utf-8"
                )
            )

            summary_rows.append(
                evaluation["summary_row"]
            )

            completed += 1
            continue

        episode_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            source_interval = (
                source_item["intervals"][0]
            )

            query_timestamp = (
                source_interval[0]
                + source_interval[1]
            ) / 2.0

            query_path = (
                episode_dir
                / "query.png"
            )

            extract_frame(
                video_path=source_item[
                    "video_path"
                ],
                timestamp_sec=query_timestamp,
                output_path=query_path,
            )

            source_suffix = (
                source_item[
                    "video_path"
                ].suffix.lower()
                or ".mp4"
            )

            target_suffix = (
                target_item[
                    "video_path"
                ].suffix.lower()
                or ".mp4"
            )

            query_full_info = (
                materialize_full_video(
                    source=source_item[
                        "video_path"
                    ],
                    destination=(
                        episode_dir
                        / (
                            "query_source_full"
                            + source_suffix
                        )
                    ),
                    mode=args.full_video_mode,
                )
            )

            target_full_info = (
                materialize_full_video(
                    source=target_item[
                        "video_path"
                    ],
                    destination=(
                        episode_dir
                        / (
                            "target_full"
                            + target_suffix
                        )
                    ),
                    mode=args.full_video_mode,
                )
            )

            search_output_dir = (
                episode_dir
                / "search"
            )

            manifest = {
                "episode_index": episode_index,
                "action_group": group_key,
                "query_text": source_item[
                    "query_text"
                ],
                "query_image": str(
                    query_path.resolve()
                ),
                "query_source_video": str(
                    source_item[
                        "video_path"
                    ]
                ),
                "query_source_interval": list(
                    source_interval
                ),
                "query_timestamp_sec": (
                    query_timestamp
                ),
                "target_video": str(
                    target_item[
                        "video_path"
                    ]
                ),
                "target_ground_truth": [
                    list(interval)
                    for interval
                    in target_item[
                        "intervals"
                    ]
                ],
                "query_full_video": (
                    query_full_info
                ),
                "target_full_video": (
                    target_full_info
                ),
            }

            (
                episode_dir
                / "pair_manifest.json"
            ).write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ),
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
                str(
                    target_item[
                        "video_path"
                    ]
                ),
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

            print()
            print("=" * 72)
            print(
                f"Episode {episode_index + 1}"
                f"/{args.episodes}"
            )
            print("=" * 72)
            print(
                "Action:",
                group_key,
            )
            print(
                "Query source:",
                source_item[
                    "video_path"
                ].name,
            )
            print(
                "Target:",
                target_item[
                    "video_path"
                ].name,
            )
            print(
                "Target GT:",
                target_item[
                    "intervals"
                ],
            )

            subprocess.run(
                command,
                check=True,
            )

            prediction_path = (
                search_output_dir
                / "predictions.json"
            )

            prediction_data = json.loads(
                prediction_path.read_text(
                    encoding="utf-8"
                )
            )

            raw_predictions = (
                prediction_data.get(
                    "predictions",
                    [],
                )
            )

            evaluated_predictions: list[
                dict[str, Any]
            ] = []

            for prediction in raw_predictions:
                interval = (
                    float(
                        prediction[
                            "start_sec"
                        ]
                    ),
                    float(
                        prediction[
                            "end_sec"
                        ]
                    ),
                )

                evaluated_predictions.append(
                    {
                        **prediction,
                        "best_gt_iou": (
                            best_gt_iou(
                                interval,
                                target_item[
                                    "intervals"
                                ],
                            )
                        ),
                    }
                )

            top1_iou = (
                evaluated_predictions[
                    0
                ]["best_gt_iou"]
                if evaluated_predictions
                else 0.0
            )

            topk_best_iou = max(
                (
                    prediction[
                        "best_gt_iou"
                    ]
                    for prediction
                    in evaluated_predictions
                ),
                default=0.0,
            )

            summary_row = {
                "episode_index": (
                    episode_index
                ),
                "action_group": group_key,
                "query_source_video": (
                    source_item[
                        "video_path"
                    ].name
                ),
                "target_video": (
                    target_item[
                        "video_path"
                    ].name
                ),
                "top1_iou": top1_iou,
                "topk_best_iou": (
                    topk_best_iou
                ),
                "top1_hit_0_3": (
                    top1_iou >= 0.3
                ),
                "top1_hit_0_5": (
                    top1_iou >= 0.5
                ),
                "topk_hit_0_3": (
                    topk_best_iou >= 0.3
                ),
                "topk_hit_0_5": (
                    topk_best_iou >= 0.5
                ),
                "episode_dir": str(
                    episode_dir.resolve()
                ),
            }

            evaluation = {
                "action_group": group_key,
                "query_text": source_item[
                    "query_text"
                ],
                "query_image": str(
                    query_path.resolve()
                ),
                "query_source_full_video": (
                    query_full_info
                ),
                "target_full_video": (
                    target_full_info
                ),
                "ground_truth_intervals": [
                    list(interval)
                    for interval
                    in target_item[
                        "intervals"
                    ]
                ],
                "top1_iou": top1_iou,
                "topk_best_iou": (
                    topk_best_iou
                ),
                "predictions": (
                    evaluated_predictions
                ),
                "summary_row": (
                    summary_row
                ),
            }

            evaluation_path.write_text(
                json.dumps(
                    evaluation,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary_rows.append(
                summary_row
            )

            completed += 1

            print(
                f"Top-1 IoU:      "
                f"{top1_iou:.4f}"
            )

            print(
                f"Top-K best IoU: "
                f"{topk_best_iou:.4f}"
            )

        except Exception as exc:
            failures += 1

            failure = {
                "episode_index": (
                    episode_index
                ),
                "action_group": group_key,
                "error": repr(exc),
                "traceback": (
                    traceback.format_exc()
                ),
            }

            (
                episode_dir
                / "failure.json"
            ).write_text(
                json.dumps(
                    failure,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"[ERROR] Episode "
                f"{episode_index}: {exc}"
            )

    successful_rows = [
        row
        for row in summary_rows
        if "top1_iou" in row
    ]

    if successful_rows:
        top1_ious = np.asarray(
            [
                row["top1_iou"]
                for row in successful_rows
            ],
            dtype=np.float64,
        )

        topk_ious = np.asarray(
            [
                row["topk_best_iou"]
                for row in successful_rows
            ],
            dtype=np.float64,
        )

        aggregate = {
            "requested_episodes": (
                args.episodes
            ),
            "completed": completed,
            "failures": failures,
            "sample_fps": (
                args.sample_fps
            ),
            "window_sec": (
                args.window_sec
            ),
            "stride_sec": (
                args.stride_sec
            ),
            "top_k": args.top_k,
            "full_video_mode": (
                args.full_video_mode
            ),
            "mean_top1_iou": float(
                np.mean(top1_ious)
            ),
            "mean_topk_best_iou": float(
                np.mean(topk_ious)
            ),
            "top1_recall_iou_0_3": float(
                np.mean(top1_ious >= 0.3)
            ),
            "top1_recall_iou_0_5": float(
                np.mean(top1_ious >= 0.5)
            ),
            "topk_recall_iou_0_3": float(
                np.mean(topk_ious >= 0.3)
            ),
            "topk_recall_iou_0_5": float(
                np.mean(topk_ious >= 0.5)
            ),
        }

    else:
        aggregate = {
            "requested_episodes": (
                args.episodes
            ),
            "completed": completed,
            "failures": failures,
        }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            aggregate,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        output_root
        / "episodes.csv",
        summary_rows,
    )

    print()
    print("=" * 72)
    print("Batch completed")
    print("=" * 72)

    for key, value in aggregate.items():
        print(
            f"{key}: {value}"
        )

    print(
        "Output:",
        output_root,
    )


if __name__ == "__main__":
    main()
