from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.services.assets import register_asset
from app.services.storage import LocalStorage, StoredFile, sha256_file


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for value in range(5):
            writer.write(np.full((24, 32, 3), value * 20, np.uint8))
    finally:
        writer.release()


def seed_assets(settings, db) -> tuple[str, str]:
    storage = LocalStorage(settings)
    query_path = storage.resolve("local", "sources/images/api-query.png")
    video_path = storage.resolve("local", "sources/videos/api-video.avi")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "orange").save(query_path)
    make_video(video_path)
    query, _ = register_asset(
        db,
        storage,
        StoredFile(
            "local",
            "sources/images/api-query.png",
            query_path,
            query_path.stat().st_size,
            sha256_file(query_path),
        ),
        "api-query.png",
        expected_kind="image",
    )
    video, _ = register_asset(
        db,
        storage,
        StoredFile(
            "local",
            "sources/videos/api-video.avi",
            video_path,
            video_path.stat().st_size,
            sha256_file(video_path),
        ),
        "api-video.avi",
        expected_kind="video",
    )
    db.commit()
    return query.id, video.id


def test_create_cancel_and_retry_job(client, settings, db) -> None:
    query_id, video_id = seed_assets(settings, db)
    created = client.post(
        "/api/jobs",
        json={
            "query_image_id": query_id,
            "target_video_id": video_id,
            "parameters": {},
        },
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]
    assert created.json()["status"] == "queued"

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    retried = client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 202
    assert retried.json()["retry_of_job_id"] == job_id
    assert retried.json()["status"] == "queued"


def test_job_validation_uses_readable_error(client, settings, db) -> None:
    query_id, _ = seed_assets(settings, db)
    response = client.post(
        "/api/jobs",
        json={
            "query_image_id": query_id,
            "target_video_id": query_id,
            "parameters": {},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VIDEO_TYPE"


def test_openapi_contains_mvp_contract(client) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/assets/upload" in paths
    assert "/api/assets/scan" in paths
    assert "/api/jobs" in paths
    assert "/api/jobs/{job_id}/result" in paths
