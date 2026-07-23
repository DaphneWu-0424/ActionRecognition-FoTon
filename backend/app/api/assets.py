from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.errors import AppError
from app.config import Settings, get_settings
from app.db.base import utc_now
from app.db.models import InferenceJob, LocalizationResult, MediaAsset
from app.db.session import get_db
from app.schemas.assets import AssetRead, ScanRequest, ScanResponse
from app.schemas.common import Page
from app.services.assets import asset_urls, register_asset
from app.services.media import kind_from_path
from app.services.storage import LocalStorage, UnsafeStoragePath, ensure_within


router = APIRouter(prefix="/assets", tags=["assets"])


def _read_asset(asset: MediaAsset) -> AssetRead:
    content_url, thumbnail_url = asset_urls(asset)
    return AssetRead(
        **{
            key: getattr(asset, key)
            for key in (
                "id",
                "kind",
                "name",
                "original_filename",
                "mime_type",
                "file_size",
                "duration_sec",
                "width",
                "height",
                "fps",
                "created_at",
            )
        },
        content_url=content_url,
        thumbnail_url=thumbnail_url,
    )


def _get_asset(db: Session, asset_id: str) -> MediaAsset:
    asset = db.get(MediaAsset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise AppError(404, "ASSET_NOT_FOUND", "Media asset was not found.")
    return asset


@router.get("", response_model=Page[AssetRead])
def list_assets(
    kind: str | None = None,
    search: str | None = None,
    include_derived: bool = False,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Page[AssetRead]:
    filters = [MediaAsset.deleted_at.is_(None)]
    if not include_derived:
        filters.append(MediaAsset.source_asset_id.is_(None))
    if kind:
        filters.append(MediaAsset.kind == kind)
    if search:
        value = f"%{search}%"
        filters.append(
            or_(
                MediaAsset.name.ilike(value),
                MediaAsset.original_filename.ilike(value),
            )
        )
    total = db.scalar(
        select(func.count()).select_from(MediaAsset).where(*filters)
    ) or 0
    assets = db.scalars(
        select(MediaAsset)
        .where(*filters)
        .order_by(MediaAsset.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return Page(
        items=[_read_asset(asset) for asset in assets],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/upload", response_model=AssetRead, status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AssetRead:
    filename = file.filename or "upload"
    kind = kind_from_path(Path(filename))
    if kind not in {"image", "video"}:
        raise AppError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Only supported image and video files can be uploaded.",
        )
    storage = LocalStorage(settings)
    stored = await storage.save_upload(file, kind)
    try:
        asset, _ = register_asset(
            db,
            storage,
            stored,
            original_filename=filename,
            expected_kind=kind,
        )
        db.commit()
        return _read_asset(asset)
    except Exception:
        stored.path.unlink(missing_ok=True)
        raise


@router.post("/scan", response_model=ScanResponse)
def scan_assets(
    body: ScanRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ScanResponse:
    if body.scan_root_id not in settings.scan_roots:
        raise AppError(
            400,
            "UNKNOWN_SCAN_ROOT",
            "The configured scan root does not exist.",
        )
    root = settings.scan_roots[body.scan_root_id]
    try:
        target = ensure_within(root / body.relative_dir, root)
    except UnsafeStoragePath as exc:
        raise AppError(
            400,
            "UNSAFE_SCAN_PATH",
            "The requested scan path is outside the configured root.",
        ) from exc
    if not target.is_dir():
        raise AppError(404, "SCAN_PATH_NOT_FOUND", "Scan directory not found.")

    storage = LocalStorage(settings)
    iterator = target.rglob("*") if body.recursive else target.glob("*")
    candidates = [
        path for path in iterator
        if path.is_file() and kind_from_path(path)
    ]
    created = skipped = failed = 0
    asset_ids: list[str] = []
    for path in candidates:
        try:
            stored = storage.register_scanned(body.scan_root_id, path)
            asset, was_created = register_asset(
                db,
                storage,
                stored,
                original_filename=path.name,
                expected_kind=kind_from_path(path),
            )
            asset_ids.append(asset.id)
            created += int(was_created)
            skipped += int(not was_created)
            db.commit()
        except Exception:
            db.rollback()
            failed += 1
    return ScanResponse(
        discovered=len(candidates),
        created=created,
        skipped=skipped,
        failed=failed,
        asset_ids=asset_ids,
    )


@router.get("/{asset_id}", response_model=AssetRead)
def get_asset(asset_id: str, db: Session = Depends(get_db)) -> AssetRead:
    return _read_asset(_get_asset(db, asset_id))


@router.delete("/{asset_id}", status_code=204)
def delete_asset(asset_id: str, db: Session = Depends(get_db)) -> None:
    asset = _get_asset(db, asset_id)
    job_reference = db.scalar(
        select(InferenceJob.id).where(
            or_(
                InferenceJob.query_image_id == asset.id,
                InferenceJob.target_video_id == asset.id,
            )
        ).limit(1)
    )
    result_reference = db.scalar(
        select(LocalizationResult.id).where(
            or_(
                LocalizationResult.clip_asset_id == asset.id,
                LocalizationResult.best_frame_asset_id == asset.id,
            )
        ).limit(1)
    )
    if job_reference or result_reference:
        raise AppError(
            409,
            "ASSET_IN_USE",
            "The asset is referenced by an inference job and cannot be deleted.",
        )
    asset.deleted_at = utc_now()
    db.commit()


@router.get("/{asset_id}/thumbnail")
def get_thumbnail(
    asset_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    asset = _get_asset(db, asset_id)
    if not asset.thumbnail_asset_id:
        raise AppError(404, "THUMBNAIL_NOT_FOUND", "Thumbnail is not available.")
    return _content_response(
        _get_asset(db, asset.thumbnail_asset_id),
        None,
        LocalStorage(settings),
    )


@router.get("/{asset_id}/content")
def get_content(
    asset_id: str,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return _content_response(
        _get_asset(db, asset_id),
        request.headers.get("range"),
        LocalStorage(settings),
    )


def _content_response(
    asset: MediaAsset,
    range_header: str | None,
    storage: LocalStorage,
):
    try:
        path = storage.resolve(asset.storage_provider, asset.storage_key)
    except UnsafeStoragePath as exc:
        raise AppError(404, "ASSET_FILE_MISSING", "Asset file is unavailable.") from exc
    if not path.is_file():
        raise AppError(404, "ASSET_FILE_MISSING", "Asset file is unavailable.")
    if not range_header:
        return FileResponse(
            path,
            media_type=asset.mime_type,
            filename=None,
            headers={"Accept-Ranges": "bytes"},
        )
    return _range_response(path, asset.mime_type, range_header)


def _range_response(path: Path, mime_type: str, header: str) -> StreamingResponse:
    size = path.stat().st_size
    try:
        unit, value = header.strip().split("=", 1)
        if unit.lower() != "bytes" or "," in value:
            raise ValueError
        start_text, end_text = value.split("-", 1)
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            start = max(0, size - suffix)
            end = size - 1
        if start < 0 or end < start or start >= size:
            raise ValueError
        end = min(end, size - 1)
    except ValueError as exc:
        raise AppError(
            416,
            "INVALID_RANGE",
            "The requested byte range is invalid.",
            {"size": size},
        ) from exc

    length = end - start + 1

    def chunks() -> Iterator[bytes]:
        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=mime_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        },
    )
