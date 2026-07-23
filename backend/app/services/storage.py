from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fastapi import UploadFile

from app.config import Settings


class UnsafeStoragePath(ValueError):
    pass


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise UnsafeStoragePath(value)
    return Path(*pure.parts)


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafeStoragePath(str(path)) from exc
    return resolved


@dataclass(frozen=True)
class StoredFile:
    provider: str
    key: str
    path: Path
    size: int
    sha256: str


class LocalStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.storage_root.resolve()

    def resolve(self, provider: str, key: str) -> Path:
        relative = _safe_relative(key)
        if provider == "local":
            return ensure_within(self.root / relative, self.root)
        if provider.startswith("scan:"):
            root_id = provider.removeprefix("scan:")
            try:
                scan_root = self.settings.scan_roots[root_id]
            except KeyError as exc:
                raise UnsafeStoragePath(f"Unknown scan root: {root_id}") from exc
            return ensure_within(scan_root / relative, scan_root)
        raise UnsafeStoragePath(f"Unknown provider: {provider}")

    def register_scanned(self, root_id: str, path: Path) -> StoredFile:
        scan_root = self.settings.scan_roots[root_id]
        resolved = ensure_within(path, scan_root)
        relative = resolved.relative_to(scan_root).as_posix()
        return StoredFile(
            provider=f"scan:{root_id}",
            key=relative,
            path=resolved,
            size=resolved.stat().st_size,
            sha256=sha256_file(resolved),
        )

    async def save_upload(
        self,
        upload: UploadFile,
        kind: str,
    ) -> StoredFile:
        suffix = Path(upload.filename or "").suffix.lower()
        relative = (
            Path("sources")
            / ("images" if kind == "image" else "videos")
            / f"{uuid.uuid4().hex}{suffix}"
        )
        destination = self.resolve("local", relative.as_posix())
        temporary = self.resolve(
            "local",
            f"tmp/{uuid.uuid4().hex}.upload",
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise ValueError("Upload exceeds configured size limit")
                    digest.update(chunk)
                    target.write(chunk)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return StoredFile(
            provider="local",
            key=relative.as_posix(),
            path=destination,
            size=size,
            sha256=digest.hexdigest(),
        )

    def import_derived(
        self,
        source: Path,
        relative: str,
    ) -> StoredFile:
        destination = self.resolve("local", relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.move(str(source), str(destination))
        return StoredFile(
            provider="local",
            key=relative,
            path=destination,
            size=destination.stat().st_size,
            sha256=sha256_file(destination),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
