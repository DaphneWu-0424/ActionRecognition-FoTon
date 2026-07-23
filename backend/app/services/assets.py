from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MediaAsset
from app.services.media import MediaInfo, create_thumbnail, probe_media
from app.services.storage import LocalStorage, StoredFile


def asset_urls(asset: MediaAsset) -> tuple[str, str | None]:
    return (
        f"/api/assets/{asset.id}/content",
        (
            f"/api/assets/{asset.thumbnail_asset_id}/content"
            if asset.thumbnail_asset_id
            else None
        ),
    )


def register_asset(
    db: Session,
    storage: LocalStorage,
    stored: StoredFile,
    original_filename: str,
    name: str | None = None,
    expected_kind: str | None = None,
    source_asset_id: str | None = None,
    create_preview: bool = True,
) -> tuple[MediaAsset, bool]:
    existing = db.scalar(
        select(MediaAsset).where(
            MediaAsset.storage_provider == stored.provider,
            MediaAsset.storage_key == stored.key,
        )
    )
    if existing:
        return existing, False

    info = probe_media(stored.path, expected_kind)
    asset = _build_asset(
        stored=stored,
        info=info,
        original_filename=original_filename,
        name=name,
        source_asset_id=source_asset_id,
    )
    db.add(asset)
    db.flush()

    if create_preview and info.kind in {"image", "video"}:
        thumbnail_relative = f"derived/thumbnails/{asset.id}.jpg"
        thumbnail_path = storage.resolve("local", thumbnail_relative)
        create_thumbnail(stored.path, info.kind, thumbnail_path)
        thumbnail_stored = StoredFile(
            provider="local",
            key=thumbnail_relative,
            path=thumbnail_path,
            size=thumbnail_path.stat().st_size,
            sha256=_sha256(thumbnail_path),
        )
        thumbnail_info = probe_media(thumbnail_path, "image")
        thumbnail = _build_asset(
            stored=thumbnail_stored,
            info=thumbnail_info,
            original_filename=f"{asset.id}.jpg",
            name=f"{asset.name} thumbnail",
            source_asset_id=asset.id,
        )
        db.add(thumbnail)
        db.flush()
        asset.thumbnail_asset_id = thumbnail.id

    return asset, True


def _build_asset(
    stored: StoredFile,
    info: MediaInfo,
    original_filename: str,
    name: str | None,
    source_asset_id: str | None,
) -> MediaAsset:
    return MediaAsset(
        id=str(uuid.uuid4()),
        kind=info.kind,
        name=name or Path(original_filename).stem,
        original_filename=original_filename,
        storage_provider=stored.provider,
        storage_key=stored.key,
        mime_type=info.mime_type,
        file_size=stored.size,
        sha256=stored.sha256,
        duration_sec=info.duration_sec,
        width=info.width,
        height=info.height,
        fps=info.fps,
        source_asset_id=source_asset_id,
        metadata_json=(
            {"frame_count": info.frame_count}
            if info.frame_count is not None
            else {}
        ),
    )


def register_derived_asset(
    db: Session,
    storage: LocalStorage,
    source: Path,
    relative: str,
    kind: str,
    source_asset_id: str,
) -> MediaAsset:
    stored = storage.import_derived(source, relative)
    asset, _ = register_asset(
        db=db,
        storage=storage,
        stored=stored,
        original_filename=Path(relative).name,
        expected_kind=kind,
        source_asset_id=source_asset_id,
        create_preview=False,
    )
    return asset


def _sha256(path: Path) -> str:
    from app.services.storage import sha256_file

    return sha256_file(path)
