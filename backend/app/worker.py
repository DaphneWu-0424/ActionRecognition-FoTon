from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utc_now
from app.db.init_db import migrate_database
from app.db.models import InferenceJob, LocalizationResult
from app.db.session import SessionLocal
from app.inference.base import InferenceCancelled, LocalizationParams
from app.inference.openclip_locator import OpenClipTemporalLocator
from app.schemas.jobs import LocalizationParameters
from app.services.assets import register_derived_asset
from app.services.jobs import claim_next_job, recover_stale_jobs
from app.services.storage import LocalStorage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("action-worker")


class Worker:
    def __init__(
        self,
        locator=None,
        settings=None,
        session_factory=SessionLocal,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory
        self.storage = LocalStorage(self.settings)
        self.locator = locator or OpenClipTemporalLocator(
            model_cache=self.settings.model_cache
        )

    def run_forever(self) -> None:
        migrate_database()
        with self.session_factory() as db:
            recovered = recover_stale_jobs(
                db, self.settings.stale_job_seconds
            )
            if recovered:
                logger.warning("Recovered %s stale jobs", recovered)
        self.locator.load()
        logger.info("Worker ready")
        while True:
            handled = self.run_once()
            if not handled:
                time.sleep(self.settings.worker_poll_seconds)

    def run_once(self) -> bool:
        with self.session_factory() as db:
            job = claim_next_job(db)
            if job is None:
                return False
            self._execute(db, job)
            return True

    def _execute(self, db: Session, job: InferenceJob) -> None:
        job_id = job.id
        temporary = self.settings.storage_root / "tmp" / job_id
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)

        def progress(stage: str, value: float) -> None:
            current = db.get(InferenceJob, job_id)
            if current is None:
                return
            current.stage = stage
            current.progress = value
            current.heartbeat_at = utc_now()
            db.commit()

        def cancelled() -> bool:
            db.expire_all()
            current = db.get(InferenceJob, job_id)
            return bool(current and current.cancel_requested_at)

        try:
            query_path = self.storage.resolve(
                job.query_image.storage_provider,
                job.query_image.storage_key,
            )
            video_path = self.storage.resolve(
                job.target_video.storage_provider,
                job.target_video.storage_key,
            )
            parameters = LocalizationParameters.model_validate(
                job.parameters_json
            ).to_domain()
            output = self.locator.localize(
                query_image=query_path,
                target_video=video_path,
                output_dir=temporary,
                params=parameters,
                on_progress=progress,
                is_cancelled=cancelled,
            )
            if cancelled():
                raise InferenceCancelled("Inference was cancelled")

            clip_asset = register_derived_asset(
                db,
                self.storage,
                output.result.clip_path,
                f"derived/jobs/{job_id}/top1_clip.mp4",
                "video",
                job.target_video_id,
            )
            frame_asset = register_derived_asset(
                db,
                self.storage,
                output.result.best_frame_path,
                f"derived/jobs/{job_id}/top1_best_frame.png",
                "image",
                job.target_video_id,
            )
            result = LocalizationResult(
                id=str(uuid.uuid4()),
                job_id=job_id,
                start_sec=output.result.start_sec,
                end_sec=output.result.end_sec,
                score=output.result.score,
                mean_score=output.result.mean_score,
                max_score=output.result.max_score,
                best_frame_sec=output.result.best_frame_sec,
                clip_asset_id=clip_asset.id,
                best_frame_asset_id=frame_asset.id,
                metadata_json={
                    "video": {
                        "native_fps": output.video.native_fps,
                        "sampled_frames": output.video.sampled_frames,
                    }
                },
            )
            db.add(result)
            current = db.get(InferenceJob, job_id)
            assert current is not None
            current.status = "succeeded"
            current.stage = "saving_results"
            current.progress = 1.0
            current.finished_at = utc_now()
            current.heartbeat_at = utc_now()
            db.commit()
            logger.info("Job %s succeeded", job_id)
        except InferenceCancelled:
            db.rollback()
            current = db.get(InferenceJob, job_id)
            if current:
                current.status = "cancelled"
                current.stage = None
                current.finished_at = utc_now()
                db.commit()
            logger.info("Job %s cancelled", job_id)
        except Exception as exc:
            db.rollback()
            error_relative = f"derived/jobs/{job_id}/error.log"
            error_path = self.storage.resolve("local", error_relative)
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            current = db.get(InferenceJob, job_id)
            if current:
                current.status = "failed"
                current.stage = None
                current.error_code = "INFERENCE_FAILED"
                current.error_message = str(exc)[:1000]
                current.error_log_key = error_relative
                current.finished_at = utc_now()
                db.commit()
            logger.exception("Job %s failed", job_id)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    Worker().run_forever()


if __name__ == "__main__":
    main()
