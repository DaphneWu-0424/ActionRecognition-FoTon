from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.errors import AppError
from app.db.models import InferenceJob, LocalizationResult, MediaAsset
from app.db.session import get_db
from app.schemas.common import Page
from app.schemas.jobs import (
    AssetSummary,
    JobCreate,
    JobListItem,
    JobRead,
    LocalizationParameters,
    ResultRead,
)
from app.services.assets import asset_urls
from app.services.jobs import create_job, request_cancel


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _asset_summary(asset: MediaAsset) -> AssetSummary:
    content_url, thumbnail_url = asset_urls(asset)
    return AssetSummary(
        id=asset.id,
        name=asset.name,
        kind=asset.kind,
        duration_sec=asset.duration_sec,
        content_url=content_url,
        thumbnail_url=thumbnail_url,
    )


def _result_read(result: LocalizationResult | None) -> ResultRead | None:
    if result is None:
        return None
    return ResultRead(
        id=result.id,
        start_sec=result.start_sec,
        end_sec=result.end_sec,
        score=result.score,
        mean_score=result.mean_score,
        max_score=result.max_score,
        best_frame_sec=result.best_frame_sec,
        clip_url=f"/api/assets/{result.clip_asset_id}/content",
        best_frame_url=f"/api/assets/{result.best_frame_asset_id}/content",
    )


def _job_options():
    return (
        selectinload(InferenceJob.query_image),
        selectinload(InferenceJob.target_video),
        selectinload(InferenceJob.result),
    )


def _get_job(db: Session, job_id: str) -> InferenceJob:
    job = db.scalar(
        select(InferenceJob)
        .where(InferenceJob.id == job_id)
        .options(*_job_options())
    )
    if job is None:
        raise AppError(404, "JOB_NOT_FOUND", "Inference job was not found.")
    return job


def _job_read(job: InferenceJob) -> JobRead:
    return JobRead(
        id=job.id,
        retry_of_job_id=job.retry_of_job_id,
        model_key=job.model_key,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        parameters=LocalizationParameters.model_validate(job.parameters_json),
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        query_image=_asset_summary(job.query_image),
        target_video=_asset_summary(job.target_video),
        result=_result_read(job.result),
    )


@router.post("", response_model=JobRead, status_code=202)
def post_job(body: JobCreate, db: Session = Depends(get_db)) -> JobRead:
    query = db.get(MediaAsset, body.query_image_id)
    video = db.get(MediaAsset, body.target_video_id)
    if query is None or query.deleted_at is not None:
        raise AppError(404, "QUERY_NOT_FOUND", "Query image was not found.")
    if video is None or video.deleted_at is not None:
        raise AppError(404, "VIDEO_NOT_FOUND", "Target video was not found.")
    if query.kind != "image":
        raise AppError(400, "INVALID_QUERY_TYPE", "Query asset must be an image.")
    if video.kind != "video":
        raise AppError(400, "INVALID_VIDEO_TYPE", "Target asset must be a video.")
    job = create_job(db, query, video, body.parameters)
    return _job_read(_get_job(db, job.id))


@router.get("", response_model=Page[JobListItem])
def list_jobs(
    status: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Page[JobListItem]:
    filters = [InferenceJob.status == status] if status else []
    total = db.scalar(
        select(func.count()).select_from(InferenceJob).where(*filters)
    ) or 0
    jobs = db.scalars(
        select(InferenceJob)
        .where(*filters)
        .options(
            selectinload(InferenceJob.query_image),
            selectinload(InferenceJob.target_video),
        )
        .order_by(InferenceJob.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return Page(
        items=[
            JobListItem(
                id=job.id,
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                created_at=job.created_at,
                finished_at=job.finished_at,
                query_image=_asset_summary(job.query_image),
                target_video=_asset_summary(job.target_video),
            )
            for job in jobs
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobRead:
    return _job_read(_get_job(db, job_id))


@router.get("/{job_id}/result", response_model=ResultRead)
def get_result(job_id: str, db: Session = Depends(get_db)) -> ResultRead:
    result = _get_job(db, job_id).result
    if result is None:
        raise AppError(404, "RESULT_NOT_READY", "Top-1 result is not available.")
    assert (read := _result_read(result)) is not None
    return read


@router.post("/{job_id}/cancel", response_model=JobRead)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> JobRead:
    request_cancel(db, _get_job(db, job_id))
    return _job_read(_get_job(db, job_id))


@router.post("/{job_id}/retry", response_model=JobRead, status_code=202)
def retry_job(job_id: str, db: Session = Depends(get_db)) -> JobRead:
    source = _get_job(db, job_id)
    if source.status not in {"failed", "cancelled", "succeeded"}:
        raise AppError(
            409,
            "JOB_NOT_RETRYABLE",
            "Only a finished job can be retried.",
        )
    job = create_job(
        db,
        source.query_image,
        source.target_video,
        LocalizationParameters.model_validate(source.parameters_json),
        retry_of_job_id=source.id,
    )
    return _job_read(_get_job(db, job.id))
