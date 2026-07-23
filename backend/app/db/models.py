from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class MediaAsset(TimestampMixin, Base):
    __tablename__ = "media_assets"
    __table_args__ = (
        UniqueConstraint(
            "storage_provider",
            "storage_key",
            name="storage_location",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_provider: Mapped[str] = mapped_column(String(80))
    storage_key: Mapped[str] = mapped_column(String(1000))
    mime_type: Mapped[str] = mapped_column(String(120))
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    thumbnail_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id")
    )
    source_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    thumbnail: Mapped["MediaAsset | None"] = relationship(
        foreign_keys=[thumbnail_asset_id],
        remote_side=[id],
        post_update=True,
    )
    source_asset: Mapped["MediaAsset | None"] = relationship(
        foreign_keys=[source_asset_id],
        remote_side=[id],
        post_update=True,
    )


class InferenceJob(TimestampMixin, Base):
    __tablename__ = "inference_jobs"
    __table_args__ = (
        CheckConstraint(
            "progress >= 0 AND progress <= 1",
            name="progress_range",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    retry_of_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("inference_jobs.id")
    )
    query_image_id: Mapped[str] = mapped_column(
        ForeignKey("media_assets.id"), index=True
    )
    target_video_id: Mapped[str] = mapped_column(
        ForeignKey("media_assets.id"), index=True
    )
    model_key: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str | None] = mapped_column(String(40))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_log_key: Mapped[str | None] = mapped_column(String(1000))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    query_image: Mapped[MediaAsset] = relationship(
        foreign_keys=[query_image_id]
    )
    target_video: Mapped[MediaAsset] = relationship(
        foreign_keys=[target_video_id]
    )
    result: Mapped["LocalizationResult | None"] = relationship(
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan",
    )


class LocalizationResult(Base):
    __tablename__ = "localization_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("inference_jobs.id"), unique=True, index=True
    )
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    mean_score: Mapped[float] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float)
    best_frame_sec: Mapped[float] = mapped_column(Float)
    clip_asset_id: Mapped[str] = mapped_column(ForeignKey("media_assets.id"))
    best_frame_asset_id: Mapped[str] = mapped_column(
        ForeignKey("media_assets.id")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    job: Mapped[InferenceJob] = relationship(back_populates="result")
    clip_asset: Mapped[MediaAsset] = relationship(
        foreign_keys=[clip_asset_id]
    )
    best_frame_asset: Mapped[MediaAsset] = relationship(
        foreign_keys=[best_frame_asset_id]
    )
