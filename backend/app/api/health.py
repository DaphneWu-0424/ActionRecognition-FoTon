from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db


router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    db.execute(text("SELECT 1"))
    writable = settings.storage_root.is_dir()
    return {
        "status": "ok" if writable else "degraded",
        "database": "ok",
        "storage": "ok" if writable else "unavailable",
        "model_cache": settings.model_cache.is_dir(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "media_fallback": "opencv",
    }
