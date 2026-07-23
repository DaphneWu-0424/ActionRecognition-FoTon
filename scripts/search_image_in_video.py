from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class Prediction:
    rank: int
    start_sec: float
    end_sec: float
    score: float
    mean_score: float
    max_score: float
    best_frame_sec: float
    interval_video: str = ""
    best_frame_image: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a video for temporal intervals visually "
            "similar to one query image using OpenCLIP."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ActionRecognition"),
    )

    parser.add_argument(
        "--query-image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Number of video frames encoded per second.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Temporal window length.",
    )

    parser.add_argument(
        "--stride-sec",
        type=float,
        default=0.5,
        help="Sliding-window stride.",
    )

    parser.add_argument(
        "--top-frame-ratio",
        type=float,
        default=0.5,
        help=(
            "Window score is the mean of its highest-scoring "
            "frames. This parameter controls the retained ratio."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.3,
        help="Temporal NMS overlap threshold.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 4),
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.root = args.root.resolve()
    args.query_image = args.query_image.resolve()
    args.video = args.video.resolve()

    if args.model_cache is None:
        args.model_cache = (
            args.root
            / "models"
            / "openclip_vit_b32_openai"
        )

    if args.output_dir is None:
        args.output_dir = (
            args.root
            / "outputs"
            / f"search_{args.video.stem}"
        )

    args.model_cache = args.model_cache.resolve()
    args.output_dir = args.output_dir.resolve()


def validate_args(args: argparse.Namespace) -> None:
    if not args.query_image.exists():
        raise FileNotFoundError(
            f"Query image not found: {args.query_image}"
        )

    if not args.video.exists():
        raise FileNotFoundError(
            f"Video not found: {args.video}"
        )

    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive.")

    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive.")

    if args.stride_sec <= 0:
        raise ValueError("--stride-sec must be positive.")

    if not 0 < args.top_frame_ratio <= 1:
        raise ValueError(
            "--top-frame-ratio must be in (0, 1]."
        )

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    if not 0 <= args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in [0, 1].")


class OpenClipEncoder:
    def __init__(
        self,
        model_cache: Path,
        batch_size: int,
        threads: int,
    ) -> None:
        torch.set_num_threads(max(1, threads))

        self.batch_size = batch_size
        self.device = "cpu"

        model_cache.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Loading OpenCLIP ViT-B-32 / OpenAI on CPU...")

        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="openai",
                device=self.device,
                cache_dir=str(model_cache),
            )
        )

        self.model = model.eval()
        self.preprocess = preprocess

    def encode_images(
        self,
        images: list[Image.Image],
    ) -> np.ndarray:
        if not images:
            raise ValueError("Image list must not be empty.")

        output: list[np.ndarray] = []

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):
            current = images[
                start : start + self.batch_size
            ]

            tensor = torch.stack(
                [
                    self.preprocess(image)
                    for image in current
                ],
                dim=0,
            )

            with torch.inference_mode():
                features = self.model.encode_image(
                    tensor,
                    normalize=True,
                )

            output.append(
                features.cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        return np.concatenate(output, axis=0)

    def encode_query(
        self,
        image_path: Path,
    ) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        feature = self.encode_images([image])[0]

        return image, feature


def flush_frame_batch(
    encoder: OpenClipEncoder,
    images: list[Image.Image],
    times: list[float],
    all_times: list[float],
    all_features: list[np.ndarray],
) -> None:
    if not images:
        return

    features = encoder.encode_images(images)

    all_times.extend(times)
    all_features.append(features)

    images.clear()
    times.clear()


def encode_sampled_video_frames(
    video_path: Path,
    encoder: OpenClipEncoder,
    sample_fps: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    native_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = 25.0

    if frame_count > 0:
        estimated_duration = frame_count / native_fps
    else:
        estimated_duration = 0.0

    sample_interval = 1.0 / sample_fps
    next_sample_sec = 0.0

    batch_images: list[Image.Image] = []
    batch_times: list[float] = []

    all_times: list[float] = []
    all_features: list[np.ndarray] = []

    frame_index = 0
    last_timestamp = 0.0

    print()
    print("Encoding sampled video frames...")
    print(f"Native FPS:       {native_fps:.3f}")
    print(f"Estimated length: {estimated_duration:.3f} sec")
    print(f"Sampling FPS:     {sample_fps:.3f}")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                break

            timestamp = float(
                cap.get(cv2.CAP_PROP_POS_MSEC)
            ) / 1000.0

            if (
                not np.isfinite(timestamp)
                or timestamp < 0
                or (
                    timestamp == 0
                    and frame_index > 0
                )
            ):
                timestamp = frame_index / native_fps

            last_timestamp = max(
                last_timestamp,
                timestamp,
            )

            if timestamp + 1e-6 >= next_sample_sec:
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                batch_images.append(
                    Image.fromarray(rgb)
                )

                batch_times.append(timestamp)

                while next_sample_sec <= timestamp + 1e-6:
                    next_sample_sec += sample_interval

                if len(batch_images) >= encoder.batch_size:
                    flush_frame_batch(
                        encoder=encoder,
                        images=batch_images,
                        times=batch_times,
                        all_times=all_times,
                        all_features=all_features,
                    )

            frame_index += 1

    finally:
        cap.release()

    flush_frame_batch(
        encoder=encoder,
        images=batch_images,
        times=batch_times,
        all_times=all_times,
        all_features=all_features,
    )

    if not all_features:
        raise RuntimeError(
            "No video frames were successfully encoded."
        )

    times_array = np.asarray(
        all_times,
        dtype=np.float32,
    )

    features_array = np.concatenate(
        all_features,
        axis=0,
    )

    duration = max(
        estimated_duration,
        last_timestamp + 1.0 / native_fps,
    )

    metadata = {
        "native_fps": native_fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "sampled_frames": len(times_array),
    }

    return times_array, features_array, metadata


def build_window_starts(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
) -> list[float]:
    if duration_sec <= window_sec:
        return [0.0]

    starts = list(
        np.arange(
            0.0,
            duration_sec - window_sec + 1e-8,
            stride_sec,
            dtype=np.float64,
        )
    )

    final_start = max(
        0.0,
        duration_sec - window_sec,
    )

    if not starts or abs(starts[-1] - final_start) > 1e-3:
        starts.append(final_start)

    return sorted(set(round(x, 6) for x in starts))


def score_windows(
    sample_times: np.ndarray,
    frame_scores: np.ndarray,
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    top_frame_ratio: float,
) -> list[dict]:
    windows: list[dict] = []

    for start_sec in build_window_starts(
        duration_sec=duration_sec,
        window_sec=window_sec,
        stride_sec=stride_sec,
    ):
        end_sec = min(
            duration_sec,
            start_sec + window_sec,
        )

        indices = np.flatnonzero(
            (sample_times >= start_sec)
            & (sample_times < end_sec)
        )

        if len(indices) == 0:
            continue

        scores = frame_scores[indices]

        keep_count = max(
            1,
            math.ceil(
                len(scores) * top_frame_ratio
            ),
        )

        top_scores = np.sort(scores)[-keep_count:]

        local_best = int(
            indices[np.argmax(scores)]
        )

        windows.append(
            {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "score": float(np.mean(top_scores)),
                "mean_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
                "best_frame_sec": float(
                    sample_times[local_best]
                ),
                "sample_count": int(len(indices)),
            }
        )

    return windows


def temporal_iou(
    first: dict,
    second: dict,
) -> float:
    intersection = max(
        0.0,
        min(first["end_sec"], second["end_sec"])
        - max(first["start_sec"], second["start_sec"]),
    )

    union = (
        first["end_sec"]
        - first["start_sec"]
        + second["end_sec"]
        - second["start_sec"]
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def temporal_nms(
    windows: list[dict],
    top_k: int,
    iou_threshold: float,
) -> list[dict]:
    candidates = sorted(
        windows,
        key=lambda item: item["score"],
        reverse=True,
    )

    selected: list[dict] = []

    for candidate in candidates:
        overlaps = [
            temporal_iou(candidate, existing)
            for existing in selected
        ]

        if all(
            overlap <= iou_threshold
            for overlap in overlaps
        ):
            selected.append(candidate)

        if len(selected) >= top_k:
            break

    return selected


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def export_interval_with_ffmpeg(
    ffmpeg_path: str,
    source_video: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    duration = max(
        0.05,
        end_sec - start_sec,
    )

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def export_interval_with_opencv(
    source_video: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    cap = cv2.VideoCapture(str(source_video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open source video: {source_video}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create output video: {output_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        start_sec * 1000.0,
    )

    try:
        while True:
            timestamp = float(
                cap.get(cv2.CAP_PROP_POS_MSEC)
            ) / 1000.0

            if timestamp >= end_sec:
                break

            ok, frame = cap.read()

            if not ok or frame is None:
                break

            writer.write(frame)

    finally:
        writer.release()
        cap.release()


def export_interval(
    source_video: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ffmpeg_path = find_ffmpeg()

    if ffmpeg_path is not None:
        try:
            export_interval_with_ffmpeg(
                ffmpeg_path=ffmpeg_path,
                source_video=source_video,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            return

        except subprocess.CalledProcessError:
            print(
                "[WARNING] FFmpeg export failed. "
                "Falling back to OpenCV."
            )

    export_interval_with_opencv(
        source_video=source_video,
        output_path=output_path,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def save_frame_at_time(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))

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
                f"Cannot read frame at {timestamp_sec:.3f}s"
            )

        if not cv2.imwrite(
            str(output_path),
            frame,
        ):
            raise RuntimeError(
                f"Cannot save image: {output_path}"
            )

    finally:
        cap.release()


def write_window_csv(
    output_path: Path,
    windows: list[dict],
) -> None:
    if not windows:
        return

    fields = [
        "start_sec",
        "end_sec",
        "score",
        "mean_score",
        "max_score",
        "best_frame_sec",
        "sample_count",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(windows)


def write_frame_csv(
    output_path: Path,
    sample_times: np.ndarray,
    frame_scores: np.ndarray,
) -> None:
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time_sec",
                "similarity",
            ],
        )

        writer.writeheader()

        for timestamp, score in zip(
            sample_times,
            frame_scores,
            strict=True,
        ):
            writer.writerow(
                {
                    "time_sec": float(timestamp),
                    "similarity": float(score),
                }
            )


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    validate_args(args)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_output_path = (
        args.output_dir / "query.png"
    )

    encoder = OpenClipEncoder(
        model_cache=args.model_cache,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    query_image, query_feature = (
        encoder.encode_query(
            args.query_image
        )
    )

    query_image.save(
        query_output_path,
        format="PNG",
    )

    (
        sample_times,
        frame_features,
        video_metadata,
    ) = encode_sampled_video_frames(
        video_path=args.video,
        encoder=encoder,
        sample_fps=args.sample_fps,
    )

    frame_scores = (
        frame_features @ query_feature
    )

    windows = score_windows(
        sample_times=sample_times,
        frame_scores=frame_scores,
        duration_sec=video_metadata[
            "duration_sec"
        ],
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        top_frame_ratio=args.top_frame_ratio,
    )

    selected_windows = temporal_nms(
        windows=windows,
        top_k=args.top_k,
        iou_threshold=args.nms_iou,
    )

    predictions: list[Prediction] = []

    print()
    print("=" * 72)
    print("Top temporal predictions")
    print("=" * 72)

    for rank, item in enumerate(
        selected_windows,
        start=1,
    ):
        interval_path = (
            args.output_dir
            / (
                f"rank_{rank:02d}"
                f"_score_{item['score']:.4f}"
                f"_start_{item['start_sec']:.2f}"
                f"_end_{item['end_sec']:.2f}.mp4"
            )
        )

        best_frame_path = (
            args.output_dir
            / f"rank_{rank:02d}_best_frame.png"
        )

        export_interval(
            source_video=args.video,
            output_path=interval_path,
            start_sec=item["start_sec"],
            end_sec=item["end_sec"],
        )

        save_frame_at_time(
            video_path=args.video,
            timestamp_sec=item["best_frame_sec"],
            output_path=best_frame_path,
        )

        prediction = Prediction(
            rank=rank,
            start_sec=item["start_sec"],
            end_sec=item["end_sec"],
            score=item["score"],
            mean_score=item["mean_score"],
            max_score=item["max_score"],
            best_frame_sec=item["best_frame_sec"],
            interval_video=str(
                interval_path.resolve()
            ),
            best_frame_image=str(
                best_frame_path.resolve()
            ),
        )

        predictions.append(prediction)

        print(
            f"Rank {rank}: "
            f"[{prediction.start_sec:.3f}, "
            f"{prediction.end_sec:.3f}) "
            f"score={prediction.score:.4f} "
            f"best_frame={prediction.best_frame_sec:.3f}s"
        )

    if predictions:
        top1_source = Path(
            predictions[0].interval_video
        )

        shutil.copy2(
            top1_source,
            args.output_dir
            / "predicted_interval.mp4",
        )

        shutil.copy2(
            Path(
                predictions[0].best_frame_image
            ),
            args.output_dir
            / "predicted_best_frame.png",
        )

    result = {
        "query_image": str(
            query_output_path.resolve()
        ),
        "input_video": str(
            args.video.resolve()
        ),
        "model": {
            "name": "ViT-B-32",
            "pretrained": "openai",
            "device": "cpu",
        },
        "parameters": {
            "sample_fps": args.sample_fps,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "top_frame_ratio": (
                args.top_frame_ratio
            ),
            "top_k": args.top_k,
            "nms_iou": args.nms_iou,
        },
        "video_metadata": video_metadata,
        "predictions": [
            asdict(prediction)
            for prediction in predictions
        ],
    }

    with (
        args.output_dir
        / "predictions.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    write_window_csv(
        output_path=(
            args.output_dir
            / "window_scores.csv"
        ),
        windows=windows,
    )

    write_frame_csv(
        output_path=(
            args.output_dir
            / "frame_scores.csv"
        ),
        sample_times=sample_times,
        frame_scores=frame_scores,
    )

    print()
    print("=" * 72)
    print("Search completed")
    print("=" * 72)
    print(f"Query:  {query_output_path}")
    print(f"Video:  {args.video}")
    print(f"Output: {args.output_dir}")

    if predictions:
        print(
            "Top-1:  "
            f"[{predictions[0].start_sec:.3f}, "
            f"{predictions[0].end_sec:.3f})"
        )


if __name__ == "__main__":
    main()