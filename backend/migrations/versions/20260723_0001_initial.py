"""Initial local web application schema.

Revision ID: 20260723_0001
Revises:
Create Date: 2026-07-23
"""

from alembic import op
import sqlalchemy as sa


revision = "20260723_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_assets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_provider", sa.String(length=80), nullable=False),
        sa.Column("storage_key", sa.String(length=1000), nullable=False),
        sa.Column("mime_type", sa.String(length=120), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("thumbnail_asset_id", sa.String(length=36), nullable=True),
        sa.Column("source_asset_id", sa.String(length=36), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_asset_id"],
            ["media_assets.id"],
            name="fk_media_assets_source_asset_id_media_assets",
        ),
        sa.ForeignKeyConstraint(
            ["thumbnail_asset_id"],
            ["media_assets.id"],
            name="fk_media_assets_thumbnail_asset_id_media_assets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_media_assets"),
        sa.UniqueConstraint(
            "storage_provider",
            "storage_key",
            name="uq_media_assets_storage_location",
        ),
    )
    op.create_index("ix_media_assets_kind", "media_assets", ["kind"])
    op.create_index("ix_media_assets_sha256", "media_assets", ["sha256"])

    op.create_table(
        "inference_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("retry_of_job_id", sa.String(length=36), nullable=True),
        sa.Column("query_image_id", sa.String(length=36), nullable=False),
        sa.Column("target_video_id", sa.String(length=36), nullable=False),
        sa.Column("model_key", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_log_key", sa.String(length=1000), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 1",
            name="ck_inference_jobs_progress_range",
        ),
        sa.ForeignKeyConstraint(
            ["query_image_id"],
            ["media_assets.id"],
            name="fk_inference_jobs_query_image_id_media_assets",
        ),
        sa.ForeignKeyConstraint(
            ["retry_of_job_id"],
            ["inference_jobs.id"],
            name="fk_inference_jobs_retry_of_job_id_inference_jobs",
        ),
        sa.ForeignKeyConstraint(
            ["target_video_id"],
            ["media_assets.id"],
            name="fk_inference_jobs_target_video_id_media_assets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_inference_jobs"),
    )
    op.create_index("ix_inference_jobs_query_image_id", "inference_jobs", ["query_image_id"])
    op.create_index("ix_inference_jobs_status", "inference_jobs", ["status"])
    op.create_index("ix_inference_jobs_target_video_id", "inference_jobs", ["target_video_id"])

    op.create_table(
        "localization_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("mean_score", sa.Float(), nullable=False),
        sa.Column("max_score", sa.Float(), nullable=False),
        sa.Column("best_frame_sec", sa.Float(), nullable=False),
        sa.Column("clip_asset_id", sa.String(length=36), nullable=False),
        sa.Column("best_frame_asset_id", sa.String(length=36), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["best_frame_asset_id"],
            ["media_assets.id"],
            name="fk_localization_results_best_frame_asset_id_media_assets",
        ),
        sa.ForeignKeyConstraint(
            ["clip_asset_id"],
            ["media_assets.id"],
            name="fk_localization_results_clip_asset_id_media_assets",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["inference_jobs.id"],
            name="fk_localization_results_job_id_inference_jobs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_localization_results"),
        sa.UniqueConstraint("job_id", name="uq_localization_results_job_id"),
    )
    op.create_index(
        "ix_localization_results_job_id",
        "localization_results",
        ["job_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_localization_results_job_id",
        table_name="localization_results",
    )
    op.drop_table("localization_results")
    op.drop_index(
        "ix_inference_jobs_target_video_id",
        table_name="inference_jobs",
    )
    op.drop_index("ix_inference_jobs_status", table_name="inference_jobs")
    op.drop_index(
        "ix_inference_jobs_query_image_id",
        table_name="inference_jobs",
    )
    op.drop_table("inference_jobs")
    op.drop_index("ix_media_assets_sha256", table_name="media_assets")
    op.drop_index("ix_media_assets_kind", table_name="media_assets")
    op.drop_table("media_assets")
