from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import utc_now
from app.db.models import InferenceJob, MediaAsset
from app.inference.openclip_locator import MODEL_KEY
from app.schemas.jobs import LocalizationParameters


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def create_job(
    db: Session,
    query: MediaAsset,
    video: MediaAsset,
    parameters: LocalizationParameters,
    retry_of_job_id: str | None = None,
) -> InferenceJob:
    job = InferenceJob(
        id=str(uuid.uuid4()),
        retry_of_job_id=retry_of_job_id,
        query_image_id=query.id,
        target_video_id=video.id,
        model_key=MODEL_KEY,
        status="queued",
        stage=None,
        progress=0.0,
        parameters_json=parameters.model_dump(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def claim_next_job(db: Session) -> InferenceJob | None:
    job_id = db.scalar(
        select(InferenceJob.id)
        .where(InferenceJob.status == "queued")
        .order_by(InferenceJob.created_at)
        .limit(1)
    )
    if job_id is None:
        return None
    now = utc_now()
    result = db.execute(
        update(InferenceJob)
        .where(
            InferenceJob.id == job_id,
            InferenceJob.status == "queued",
        )
        .values(
            status="running",
            stage="probing_video",
            progress=0.0,
            started_at=now,
            heartbeat_at=now,
        )
    )
    db.commit()
    if result.rowcount != 1:
        return None
    return db.get(InferenceJob, job_id)


def request_cancel(db: Session, job: InferenceJob) -> None:
    if job.status in TERMINAL_STATUSES:
        return
    now = utc_now()
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = now
        job.progress = 0.0
    else:
        job.cancel_requested_at = now
    db.commit()


def recover_stale_jobs(db: Session, stale_seconds: int) -> int:
    cutoff = utc_now() - timedelta(seconds=stale_seconds)
    result = db.execute(
        update(InferenceJob)
        .where(
            InferenceJob.status == "running",
            InferenceJob.heartbeat_at < cutoff,
        )
        .values(
            status="failed",
            stage=None,
            error_code="WORKER_LOST",
            error_message="The inference worker stopped unexpectedly.",
            finished_at=utc_now(),
        )
    )
    db.commit()
    return result.rowcount
