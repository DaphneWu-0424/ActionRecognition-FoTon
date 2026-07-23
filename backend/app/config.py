from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACTION_APP_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    database_url: str = (
        f"sqlite:///{(PROJECT_ROOT / 'data/app_storage/app.sqlite3').as_posix()}"
    )
    storage_root: Path = PROJECT_ROOT / "data" / "app_storage"
    model_cache: Path = PROJECT_ROOT / "models" / "openclip_vit_b32_openai"
    scan_roots: dict[str, Path] = Field(
        default_factory=lambda: {
            "inputs": PROJECT_ROOT / "inputs",
            "thumos14": (
                PROJECT_ROOT
                / "data"
                / "etbench_thumos14"
                / "videos"
                / "thumos14"
            ),
        }
    )
    worker_poll_seconds: float = 1.0
    stale_job_seconds: int = 300
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    api_host: str = "127.0.0.1"
    api_port: int = 8010

    @field_validator("scan_roots", mode="before")
    @classmethod
    def parse_scan_roots(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("storage_root", "model_cache", mode="after")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("scan_roots", mode="after")
    @classmethod
    def resolve_scan_roots(cls, value: dict[str, Path]) -> dict[str, Path]:
        return {
            key: path.expanduser().resolve()
            for key, path in value.items()
        }

    def ensure_directories(self) -> None:
        for relative in (
            "sources/images",
            "sources/videos",
            "derived/thumbnails",
            "derived/jobs",
            "tmp",
        ):
            (self.storage_root / relative).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
