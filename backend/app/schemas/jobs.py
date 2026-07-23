from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.inference.base import LocalizationParams


class LocalizationParameters(BaseModel):
    sample_fps: float = Field(default=2.0, gt=0, le=30)
    window_sec: float = Field(default=2.0, gt=0, le=300)
    stride_sec: float = Field(default=0.5, gt=0, le=300)
    top_frame_ratio: float = Field(default=0.5, gt=0, le=1)
    nms_iou: float = Field(default=0.3, ge=0, le=1)

    def to_domain(self) -> LocalizationParams:
        return LocalizationParams(
            **self.model_dump(),
            top_k=1,
        )


class JobCreate(BaseModel):
    query_image_id: str
    target_video_id: str
    parameters: LocalizationParameters = LocalizationParameters()


class AssetSummary(BaseModel):
    id: str
    name: str
    kind: str
    duration_sec: float | None
    content_url: str
    thumbnail_url: str | None


class ResultRead(BaseModel):
    id: str
    start_sec: float
    end_sec: float
    score: float
    mean_score: float
    max_score: float
    best_frame_sec: float
    clip_url: str
    best_frame_url: str


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    retry_of_job_id: str | None
    model_key: str
    status: str
    stage: str | None
    progress: float
    parameters: LocalizationParameters
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    query_image: AssetSummary
    target_video: AssetSummary
    result: ResultRead | None


class JobListItem(BaseModel):
    id: str
    status: str
    stage: str | None
    progress: float
    created_at: datetime
    finished_at: datetime | None
    query_image: AssetSummary
    target_video: AssetSummary
