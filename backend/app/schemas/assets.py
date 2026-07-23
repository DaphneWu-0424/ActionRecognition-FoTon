from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    name: str
    original_filename: str
    mime_type: str
    file_size: int
    duration_sec: float | None
    width: int | None
    height: int | None
    fps: float | None
    created_at: datetime
    content_url: str
    thumbnail_url: str | None


class ScanRequest(BaseModel):
    scan_root_id: str
    relative_dir: str = ""
    recursive: bool = True


class ScanResponse(BaseModel):
    discovered: int
    created: int
    skipped: int
    failed: int
    asset_ids: list[str]


class AssetQuery(BaseModel):
    kind: str | None = None
    search: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
