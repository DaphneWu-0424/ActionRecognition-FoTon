from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.db.models import InferenceJob, LocalizationResult
from app.inference.base import (
    InferenceCancelled,
    LocalizationOutput,
    LocalizationResult as DomainResult,
    VideoMetadata,
)
from app.schemas.jobs import LocalizationParameters
from app.services.assets import register_asset
from app.services.jobs import create_job
from app.services.storage import LocalStorage, StoredFile, sha256_file
from app.worker import Worker


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for value in range(20):
            writer.write(
                np.full((48, 64, 3), value * 8, dtype=np.uint8)
            )
    finally:
        writer.release()


class FakeLocator:
    def load(self) -> None:
        return None

    def localize(
        self,
        query_image,
        target_video,
        output_dir,
        params,
        on_progress,
        is_cancelled,
    ) -> LocalizationOutput:
        del params
        assert not is_cancelled()
        on_progress("encoding_frames", 0.5)
        output_dir.mkdir(parents=True, exist_ok=True)
        clip = output_dir / "top1_clip.mp4"
        frame = output_dir / "top1_best_frame.png"
        shutil.copy2(target_video, clip)
        Image.open(query_image).save(frame)
        query_copy = output_dir / "query.png"
        Image.open(query_image).save(query_copy)
        frame_scores = output_dir / "frame_scores.csv"
        window_scores = output_dir / "window_scores.csv"
        frame_scores.write_text("time_sec,similarity\n", encoding="utf-8")
        window_scores.write_text("start_sec,end_sec,score\n", encoding="utf-8")
        return LocalizationOutput(
            result=DomainResult(
                start_sec=0.5,
                end_sec=1.5,
                score=0.9,
                mean_score=0.8,
                max_score=0.95,
                best_frame_sec=1.0,
                clip_path=clip,
                best_frame_path=frame,
            ),
            video=VideoMetadata(
                native_fps=10,
                frame_count=20,
                width=64,
                height=48,
                duration_sec=2,
                sampled_frames=4,
            ),
            query_copy_path=query_copy,
            frame_scores_path=frame_scores,
            window_scores_path=window_scores,
        )


class FailingLocator(FakeLocator):
    def localize(self, *args, **kwargs):
        raise RuntimeError("synthetic model failure")


class CancelledLocator(FakeLocator):
    def localize(self, *args, **kwargs):
        raise InferenceCancelled("synthetic cancellation")


def seed_job(settings, session_factory) -> str:
    storage = LocalStorage(settings)
    query_path = storage.resolve("local", "sources/images/query.png")
    video_path = storage.resolve("local", "sources/videos/target.avi")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not query_path.exists():
        Image.new("RGB", (32, 24), (30, 180, 90)).save(query_path)
    if not video_path.exists():
        make_video(video_path)
    with session_factory() as db:
        query, _ = register_asset(
            db,
            storage,
            StoredFile(
                "local",
                "sources/images/query.png",
                query_path,
                query_path.stat().st_size,
                sha256_file(query_path),
            ),
            "query.png",
            expected_kind="image",
        )
        video, _ = register_asset(
            db,
            storage,
            StoredFile(
                "local",
                "sources/videos/target.avi",
                video_path,
                video_path.stat().st_size,
                sha256_file(video_path),
            ),
            "target.avi",
            expected_kind="video",
        )
        db.commit()
        return create_job(
            db,
            query,
            video,
            LocalizationParameters(),
        ).id


def test_worker_persists_top1_result(settings, session_factory) -> None:
    storage = LocalStorage(settings)
    query_path = storage.resolve("local", "sources/images/query.png")
    video_path = storage.resolve("local", "sources/videos/target.avi")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (30, 180, 90)).save(query_path)
    make_video(video_path)

    with session_factory() as db:
        query, _ = register_asset(
            db,
            storage,
            StoredFile(
                "local",
                "sources/images/query.png",
                query_path,
                query_path.stat().st_size,
                sha256_file(query_path),
            ),
            "query.png",
            expected_kind="image",
        )
        video, _ = register_asset(
            db,
            storage,
            StoredFile(
                "local",
                "sources/videos/target.avi",
                video_path,
                video_path.stat().st_size,
                sha256_file(video_path),
            ),
            "target.avi",
            expected_kind="video",
        )
        db.commit()
        job = create_job(
            db,
            query,
            video,
            LocalizationParameters(),
        )
        job_id = job.id

    worker = Worker(
        locator=FakeLocator(),
        settings=settings,
        session_factory=session_factory,
    )
    assert worker.run_once() is True

    with session_factory() as db:
        job = db.get(InferenceJob, job_id)
        result = db.query(LocalizationResult).filter_by(job_id=job_id).one()
        assert job is not None
        assert job.status == "succeeded"
        assert result.start_sec == 0.5
        assert storage.resolve(
            result.clip_asset.storage_provider,
            result.clip_asset.storage_key,
        ).is_file()


def test_worker_records_failure_and_can_process_next_job(
    settings,
    session_factory,
) -> None:
    first_id = seed_job(settings, session_factory)
    second_id = seed_job(settings, session_factory)
    failing = Worker(
        locator=FailingLocator(),
        settings=settings,
        session_factory=session_factory,
    )
    assert failing.run_once()
    with session_factory() as db:
        first = db.get(InferenceJob, first_id)
        assert first is not None
        assert first.status == "failed"
        assert first.error_code == "INFERENCE_FAILED"

    succeeding = Worker(
        locator=FakeLocator(),
        settings=settings,
        session_factory=session_factory,
    )
    assert succeeding.run_once()
    with session_factory() as db:
        second = db.get(InferenceJob, second_id)
        assert second is not None
        assert second.status == "succeeded"


def test_worker_records_cooperative_cancellation(
    settings,
    session_factory,
) -> None:
    job_id = seed_job(settings, session_factory)
    worker = Worker(
        locator=CancelledLocator(),
        settings=settings,
        session_factory=session_factory,
    )
    assert worker.run_once()
    with session_factory() as db:
        job = db.get(InferenceJob, job_id)
        assert job is not None
        assert job.status == "cancelled"
