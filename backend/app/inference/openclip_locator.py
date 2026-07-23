from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from scripts.search_image_in_video import (
    OpenClipEncoder,
    encode_sampled_video_frames,
    export_interval,
    save_frame_at_time,
    score_windows,
    temporal_nms,
    write_frame_csv,
    write_window_csv,
)

from .base import (
    CancelCheck,
    InferenceCancelled,
    LocalizationOutput,
    LocalizationParams,
    LocalizationResult,
    ProgressCallback,
    VideoMetadata,
    never_cancelled,
    noop_progress,
)


MODEL_KEY = "openclip-vit-b32-openai"


class OpenClipTemporalLocator:
    def __init__(
        self,
        model_cache: Path,
        batch_size: int = 16,
        threads: int | None = None,
    ) -> None:
        self.model_cache = model_cache.resolve()
        self.batch_size = batch_size
        self.threads = threads or min(8, os.cpu_count() or 4)
        self._encoder: OpenClipEncoder | None = None

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    def load(self) -> None:
        if self._encoder is None:
            self._encoder = OpenClipEncoder(
                model_cache=self.model_cache,
                batch_size=self.batch_size,
                threads=self.threads,
            )

    def _check_cancelled(self, check: CancelCheck) -> None:
        if check():
            raise InferenceCancelled("Inference was cancelled")

    def localize(
        self,
        query_image: Path,
        target_video: Path,
        output_dir: Path,
        params: LocalizationParams,
        on_progress: ProgressCallback = noop_progress,
        is_cancelled: CancelCheck = never_cancelled,
    ) -> LocalizationOutput:
        params.validate()
        query_image = query_image.resolve()
        target_video = target_video.resolve()
        output_dir = output_dir.resolve()

        if not query_image.is_file():
            raise FileNotFoundError(query_image)
        if not target_video.is_file():
            raise FileNotFoundError(target_video)

        output_dir.mkdir(parents=True, exist_ok=True)
        self._check_cancelled(is_cancelled)
        self.load()
        assert self._encoder is not None

        on_progress("encoding_query", 0.05)
        query, query_feature = self._encoder.encode_query(query_image)
        query_copy_path = output_dir / "query.png"
        query.save(query_copy_path, format="PNG")

        self._check_cancelled(is_cancelled)
        on_progress("encoding_frames", 0.10)
        sample_times, frame_features, raw_metadata = (
            encode_sampled_video_frames(
                video_path=target_video,
                encoder=self._encoder,
                sample_fps=params.sample_fps,
                on_batch_progress=lambda fraction: on_progress(
                    "encoding_frames",
                    0.10 + 0.65 * fraction,
                ),
                cancel_check=lambda: self._check_cancelled(is_cancelled),
            )
        )
        on_progress("encoding_frames", 0.75)

        self._check_cancelled(is_cancelled)
        on_progress("scoring_windows", 0.78)
        frame_scores = frame_features @ query_feature
        windows = score_windows(
            sample_times=sample_times,
            frame_scores=frame_scores,
            duration_sec=raw_metadata["duration_sec"],
            window_sec=params.window_sec,
            stride_sec=params.stride_sec,
            top_frame_ratio=params.top_frame_ratio,
        )

        on_progress("selecting_results", 0.87)
        selected = temporal_nms(
            windows=windows,
            top_k=1,
            iou_threshold=params.nms_iou,
        )
        if not selected:
            raise RuntimeError("No temporal result was produced")
        best = selected[0]

        self._check_cancelled(is_cancelled)
        on_progress("exporting_results", 0.91)
        clip_path = output_dir / "top1_clip.mp4"
        best_frame_path = output_dir / "top1_best_frame.png"
        export_interval(
            source_video=target_video,
            output_path=clip_path,
            start_sec=best["start_sec"],
            end_sec=best["end_sec"],
        )
        save_frame_at_time(
            video_path=target_video,
            timestamp_sec=best["best_frame_sec"],
            output_path=best_frame_path,
        )

        frame_scores_path = output_dir / "frame_scores.csv"
        window_scores_path = output_dir / "window_scores.csv"
        write_frame_csv(
            output_path=frame_scores_path,
            sample_times=sample_times,
            frame_scores=frame_scores,
        )
        write_window_csv(
            output_path=window_scores_path,
            windows=windows,
        )

        result = LocalizationResult(
            start_sec=float(best["start_sec"]),
            end_sec=float(best["end_sec"]),
            score=float(best["score"]),
            mean_score=float(best["mean_score"]),
            max_score=float(best["max_score"]),
            best_frame_sec=float(best["best_frame_sec"]),
            clip_path=clip_path,
            best_frame_path=best_frame_path,
        )
        video = VideoMetadata(
            native_fps=float(raw_metadata["native_fps"]),
            frame_count=int(raw_metadata["frame_count"]),
            width=int(raw_metadata["width"]),
            height=int(raw_metadata["height"]),
            duration_sec=float(raw_metadata["duration_sec"]),
            sampled_frames=int(raw_metadata["sampled_frames"]),
        )
        output = LocalizationOutput(
            result=result,
            video=video,
            query_copy_path=query_copy_path,
            frame_scores_path=frame_scores_path,
            window_scores_path=window_scores_path,
        )

        manifest = {
            "query_image": str(query_copy_path),
            "input_video": str(target_video),
            "model": {
                "key": MODEL_KEY,
                "name": "ViT-B-32",
                "pretrained": "openai",
                "device": "cpu",
            },
            "parameters": asdict(params),
            "video_metadata": asdict(video),
            "result": {
                **asdict(result),
                "clip_path": str(result.clip_path),
                "best_frame_path": str(result.best_frame_path),
            },
        }
        (output_dir / "result.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        on_progress("saving_results", 1.0)
        return output
