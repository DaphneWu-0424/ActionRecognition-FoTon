from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
import shutil


VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mkv",
    ".mov",
    ".mpeg",
    ".mpg",
}

DEFAULT_CLASSES = [
    "clap",
    "drink",
    "pick",
    "pour",
    "push",
    "sit",
    "stand",
    "throw",
    "turn",
    "wave",
]


@dataclass(frozen=True)
class GalleryItem:
    segment_index: int
    action_class: str
    video_path: str
    start_sec: float
    end_sec: float
    is_positive: bool
    score: float
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HMDB51 single-image to video-clip retrieval "
            "using OpenCLIP on CPU."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ActionRecognition"),
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="HMDB51 videos root. Defaults to ROOT/data/hmdb51/videos.",
    )

    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
        help="OpenCLIP model cache directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of retrieval episodes.",
    )

    parser.add_argument(
        "--gallery-size",
        type=int,
        default=10,
        help="Number of candidate clips per episode.",
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Uniformly sampled frames used for each video embedding.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=30,
        help="Limit videos scanned per class for a fast CPU test.",
    )

    parser.add_argument(
        "--virtual-segment-sec",
        type=float,
        default=4.0,
        help="Duration assigned to each clip on the virtual timeline.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 4),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.root = args.root.resolve()

    if args.videos_dir is None:
        args.videos_dir = (
            args.root / "data" / "hmdb51" / "videos"
        )

    if args.model_cache is None:
        args.model_cache = (
            args.root / "models" / "openclip_vit_b32_openai"
        )

    if args.output_dir is None:
        args.output_dir = (
            args.root / "outputs" / "hmdb51_clip_retrieval"
        )

    args.videos_dir = args.videos_dir.resolve()
    args.model_cache = args.model_cache.resolve()
    args.output_dir = args.output_dir.resolve()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length feature vector.")

    return vector / norm


def collect_class_videos(
    videos_root: Path,
    class_names: list[str],
    max_videos_per_class: int,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    result: dict[str, list[Path]] = {}

    for class_name in class_names:
        class_dir = videos_root / class_name

        if not class_dir.exists():
            print(
                f"[WARNING] Class directory does not exist: "
                f"{class_dir}"
            )
            continue

        videos = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        )

        rng.shuffle(videos)
        videos = videos[:max_videos_per_class]

        if len(videos) < 2:
            print(
                f"[WARNING] Need at least two videos for "
                f"class {class_name}, found {len(videos)}."
            )
            continue

        result[class_name] = videos

    if len(result) < 2:
        raise RuntimeError(
            "At least two valid action classes are required. "
            f"Found: {sorted(result)}"
        )

    return result


def read_middle_frame(video_path: Path) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if frame_count <= 0:
            raise RuntimeError(
                f"Invalid frame count for {video_path}: "
                f"{frame_count}"
            )

        middle_index = max(0, frame_count // 2)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            middle_index,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                f"Cannot read middle frame from {video_path}"
            )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        return Image.fromarray(frame_rgb)

    finally:
        cap.release()

def export_video_as_mp4(
    source_path: Path,
    output_path: Path,
) -> Path:
    """
    将候选短视频统一转码为 MP4，方便直接人工查看。

    如果 OpenCV 无法创建 MP4，则保留原格式复制件。
    返回实际生成的文件路径。
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video for export: {source_path}"
        )

    writer = None

    try:
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

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid video dimensions: "
                f"{width}x{height}, source={source_path}"
            )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Cannot create MP4 writer: {output_path}"
            )

        written_frames = 0

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                break

            writer.write(frame)
            written_frames += 1

        if written_frames == 0:
            raise RuntimeError(
                f"No frames exported from {source_path}"
            )

        return output_path

    except Exception as exc:
        print(
            f"[WARNING] MP4 export failed for "
            f"{source_path}: {exc}"
        )

        fallback_path = (
            output_path.parent
            / (
                output_path.stem
                + "_original"
                + source_path.suffix.lower()
            )
        )

        shutil.copy2(
            source_path,
            fallback_path,
        )

        print(
            f"[WARNING] Original video copied to: "
            f"{fallback_path}"
        )

        return fallback_path

    finally:
        if writer is not None:
            writer.release()

        cap.release()


def sample_video_frames(
    video_path: Path,
    num_frames: int,
) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[Image.Image] = []

    try:
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if frame_count <= 0:
            raise RuntimeError(
                f"Invalid frame count for {video_path}: "
                f"{frame_count}"
            )

        indices = np.linspace(
            0,
            max(0, frame_count - 1),
            num=num_frames,
            dtype=np.int64,
        )

        for frame_index in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(
                Image.fromarray(frame_rgb)
            )

    finally:
        cap.release()

    minimum_frames = max(2, num_frames // 2)

    if len(frames) < minimum_frames:
        raise RuntimeError(
            f"Too few decodable frames from {video_path}: "
            f"{len(frames)}/{num_frames}"
        )

    return frames


class OpenClipVideoRetriever:
    def __init__(
        self,
        model_cache: Path,
        embedding_cache: Path,
        frames_per_video: int,
        batch_size: int,
        threads: int,
    ) -> None:
        self.model_name = "ViT-B-32"
        self.pretrained = "openai"
        self.frames_per_video = frames_per_video
        self.batch_size = batch_size

        model_cache.mkdir(
            parents=True,
            exist_ok=True,
        )

        embedding_cache.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_cache = embedding_cache

        torch.set_num_threads(max(1, threads))

        print(
            f"Loading {self.model_name} / "
            f"{self.pretrained} on CPU..."
        )

        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device="cpu",
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
            raise ValueError("Image batch must not be empty.")

        all_features: list[np.ndarray] = []

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):
            batch_images = images[
                start : start + self.batch_size
            ]

            batch_tensor = torch.stack(
                [
                    self.preprocess(image)
                    for image in batch_images
                ],
                dim=0,
            )

            with torch.inference_mode():
                features = self.model.encode_image(
                    batch_tensor,
                    normalize=True,
                )

            all_features.append(
                features.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

        return np.concatenate(
            all_features,
            axis=0,
        )

    def encode_query_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        feature = self.encode_images([image])[0]
        return l2_normalize(feature)

    def _video_cache_key(
        self,
        video_path: Path,
    ) -> str:
        stat = video_path.stat()

        identity = "|".join(
            [
                str(video_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                self.model_name,
                self.pretrained,
                str(self.frames_per_video),
            ]
        )

        return hashlib.sha1(
            identity.encode("utf-8")
        ).hexdigest()

    def encode_video(
        self,
        video_path: Path,
    ) -> np.ndarray:
        cache_key = self._video_cache_key(
            video_path
        )

        cache_path = (
            self.embedding_cache /
            f"{cache_key}.npy"
        )

        if cache_path.exists():
            feature = np.load(cache_path)
            return l2_normalize(
                feature.astype(
                    np.float32,
                    copy=False,
                )
            )

        frames = sample_video_frames(
            video_path=video_path,
            num_frames=self.frames_per_video,
        )

        frame_features = self.encode_images(frames)

        # Baseline video representation:
        # mean pooling over normalized frame embeddings.
        video_feature = np.mean(
            frame_features,
            axis=0,
        )

        video_feature = l2_normalize(
            video_feature.astype(
                np.float32,
                copy=False,
            )
        )

        np.save(
            cache_path,
            video_feature,
        )

        return video_feature


def select_distractors(
    class_videos: dict[str, list[Path]],
    target_class: str,
    count: int,
    rng: random.Random,
) -> list[tuple[str, Path]]:
    other_classes = [
        class_name
        for class_name in class_videos
        if class_name != target_class
    ]

    if not other_classes:
        raise RuntimeError(
            "No distractor classes are available."
        )

    selected: list[tuple[str, Path]] = []
    used_paths: set[Path] = set()

    while len(selected) < count:
        rng.shuffle(other_classes)

        for class_name in other_classes:
            candidates = [
                path
                for path in class_videos[class_name]
                if path not in used_paths
            ]

            if not candidates:
                continue

            video_path = rng.choice(candidates)
            used_paths.add(video_path)

            selected.append(
                (class_name, video_path)
            )

            if len(selected) >= count:
                break

        if not selected:
            raise RuntimeError(
                "Unable to choose distractor videos."
            )

    return selected


def write_csv(
    output_path: Path,
    items: Iterable[GalleryItem],
) -> None:
    rows = list(items)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                asdict(rows[0]).keys()
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                asdict(row)
            )


def run_episode(
    episode_index: int,
    class_videos: dict[str, list[Path]],
    retriever: OpenClipVideoRetriever,
    gallery_size: int,
    virtual_segment_sec: float,
    output_dir: Path,
    rng: random.Random,
) -> dict:
    target_class = rng.choice(
        sorted(class_videos)
    )

    query_video, positive_video = rng.sample(
        class_videos[target_class],
        k=2,
    )

    distractors = select_distractors(
        class_videos=class_videos,
        target_class=target_class,
        count=gallery_size - 1,
        rng=rng,
    )

    gallery: list[tuple[str, Path, bool]] = [
        (
            target_class,
            positive_video,
            True,
        )
    ]

    gallery.extend(
        (
            class_name,
            video_path,
            False,
        )
        for class_name, video_path
        in distractors
    )

    rng.shuffle(gallery)

    episode_dir = (
        output_dir /
        f"episode_{episode_index:03d}"
    )

    episode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_image = read_middle_frame(
        query_video
    )

    query_path = episode_dir / "query.png"

    query_image.save(
        query_path,
        format="PNG",
    )

    query_feature = (
        retriever.encode_query_image(
            query_image
        )
    )

    scored: list[dict] = []

    for segment_index, (
        action_class,
        video_path,
        is_positive,
    ) in enumerate(gallery):
        video_feature = (
            retriever.encode_video(
                video_path
            )
        )

        score = float(
            np.dot(
                query_feature,
                video_feature,
            )
        )

        scored.append(
            {
                "segment_index": segment_index,
                "action_class": action_class,
                "video_path": str(
                    video_path.resolve()
                ),
                "start_sec": (
                    segment_index *
                    virtual_segment_sec
                ),
                "end_sec": (
                    (segment_index + 1) *
                    virtual_segment_sec
                ),
                "is_positive": is_positive,
                "score": score,
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: item["score"],
        reverse=True,
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):
        item["rank"] = rank

    gallery_items = [
        GalleryItem(**item)
        for item in ranked
    ]

    positive_item = next(
        item
        for item in gallery_items
        if item.is_positive
    )

    predicted_item = gallery_items[0]

    predicted_source_path = Path(
        predicted_item.video_path
    )

    ground_truth_source_path = Path(
        positive_item.video_path
    )

    predicted_clip_path = export_video_as_mp4(
        source_path=predicted_source_path,
        output_path=(
            episode_dir
            / "predicted_interval.mp4"
        ),
    )

    ground_truth_clip_path = export_video_as_mp4(
        source_path=ground_truth_source_path,
        output_path=(
            episode_dir
            / "ground_truth_interval.mp4"
        ),
    )

    top1_correct = (
        positive_item.rank == 1
    )

    top5_correct = (
        positive_item.rank <=
        min(5, gallery_size)
    )

    episode_result = {
        "episode_index": episode_index,
        "target_class": target_class,
        "query_video": str(
            query_video.resolve()
        ),
        "query_image": str(
            query_path.resolve()
        ),
        "positive_gallery_video": str(
            positive_video.resolve()
        ),
        "positive_rank": (
            positive_item.rank
        ),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "reciprocal_rank": (
            1.0 /
            positive_item.rank
        ),
        "predicted_start_sec": (
            gallery_items[0].start_sec
        ),
        "predicted_end_sec": (
            gallery_items[0].end_sec
        ),
        "ground_truth_start_sec": (
            positive_item.start_sec
        ),
        "ground_truth_end_sec": (
            positive_item.end_sec
        ),
        "predicted_action_class": (
            predicted_item.action_class
        ),
        "predicted_score": (
            predicted_item.score
        ),
        "predicted_source_video": (
            predicted_item.video_path
        ),
        "predicted_interval_video": str(
            predicted_clip_path.resolve()
        ),
        "ground_truth_interval_video": str(
            ground_truth_clip_path.resolve()
        ),
        "gallery": [
            asdict(item)
            for item in gallery_items
        ],
    }

    with (
        episode_dir /
        "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            episode_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    write_csv(
        episode_dir / "scores.csv",
        gallery_items,
    )

    print()
    print(
        f"Episode {episode_index:03d} | "
        f"query={target_class} | "
        f"positive rank={positive_item.rank}"
    )

    for item in gallery_items[:5]:
        marker = " <-- GT" if (
            item.is_positive
        ) else ""

        print(
            f"  Top {item.rank}: "
            f"{item.action_class:12s} "
            f"score={item.score:.4f} "
            f"time=[{item.start_sec:.1f}, "
            f"{item.end_sec:.1f})"
            f"{marker}"
        )

    return episode_result


def main() -> None:
    args = parse_args()
    resolve_paths(args)

    if args.gallery_size < 2:
        raise ValueError(
            "--gallery-size must be at least 2."
        )

    if args.frames_per_video < 1:
        raise ValueError(
            "--frames-per-video must be positive."
        )

    if not args.videos_dir.exists():
        raise FileNotFoundError(
            f"HMDB51 video directory not found: "
            f"{args.videos_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_cache = (
        args.root /
        "cache" /
        "hmdb51_clip_embeddings"
    )

    class_videos = collect_class_videos(
        videos_root=args.videos_dir,
        class_names=args.classes,
        max_videos_per_class=(
            args.max_videos_per_class
        ),
        seed=args.seed,
    )

    valid_classes = sorted(class_videos)

    if args.gallery_size > len(valid_classes):
        raise ValueError(
            "--gallery-size cannot exceed the "
            "number of valid classes when using "
            "one candidate per action class. "
            f"gallery_size={args.gallery_size}, "
            f"classes={len(valid_classes)}"
        )

    print("Valid classes:")
    for class_name in valid_classes:
        print(
            f"  {class_name}: "
            f"{len(class_videos[class_name])} videos"
        )

    retriever = OpenClipVideoRetriever(
        model_cache=args.model_cache,
        embedding_cache=embedding_cache,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    rng = random.Random(args.seed)

    results: list[dict] = []

    for episode_index in tqdm(
        range(args.episodes),
        desc="Retrieval episodes",
    ):
        result = run_episode(
            episode_index=episode_index,
            class_videos=class_videos,
            retriever=retriever,
            gallery_size=args.gallery_size,
            virtual_segment_sec=(
                args.virtual_segment_sec
            ),
            output_dir=args.output_dir,
            rng=rng,
        )

        results.append(result)

    recall_at_1 = float(
        np.mean(
            [
                item["top1_correct"]
                for item in results
            ]
        )
    )

    recall_at_5 = float(
        np.mean(
            [
                item["top5_correct"]
                for item in results
            ]
        )
    )

    mean_reciprocal_rank = float(
        np.mean(
            [
                item["reciprocal_rank"]
                for item in results
            ]
        )
    )

    mean_positive_rank = float(
        np.mean(
            [
                item["positive_rank"]
                for item in results
            ]
        )
    )

    summary = {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "device": "cpu",
        "episodes": args.episodes,
        "gallery_size": args.gallery_size,
        "frames_per_video": (
            args.frames_per_video
        ),
        "classes": valid_classes,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mean_reciprocal_rank": (
            mean_reciprocal_rank
        ),
        "mean_positive_rank": (
            mean_positive_rank
        ),
    }

    with (
        args.output_dir /
        "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("Final result")
    print("=" * 72)
    print(
        f"Episodes:          "
        f"{args.episodes}"
    )
    print(
        f"Gallery size:      "
        f"{args.gallery_size}"
    )
    print(
        f"Recall@1:          "
        f"{recall_at_1:.4f}"
    )
    print(
        f"Recall@5:          "
        f"{recall_at_5:.4f}"
    )
    print(
        f"MRR:               "
        f"{mean_reciprocal_rank:.4f}"
    )
    print(
        f"Mean positive rank:"
        f" {mean_positive_rank:.3f}"
    )
    print(
        f"Output:            "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()