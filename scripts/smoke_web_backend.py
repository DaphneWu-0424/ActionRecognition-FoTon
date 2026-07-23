from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app
from app.worker import Worker


def find_asset(client: TestClient, kind: str, filename: str) -> dict:
    response = client.get(
        "/api/assets",
        params={"kind": kind, "search": filename, "page_size": 100},
    )
    response.raise_for_status()
    for item in response.json()["items"]:
        if item["original_filename"] == filename:
            return item
    raise RuntimeError(f"Asset not found after scan: {filename}")


def main() -> None:
    with TestClient(app) as client:
        scan = client.post(
            "/api/assets/scan",
            json={
                "scan_root_id": "inputs",
                "relative_dir": "",
                "recursive": True,
            },
        )
        scan.raise_for_status()
        query = find_asset(client, "image", "query.png")
        video = find_asset(client, "video", "test_video.mp4")
        created = client.post(
            "/api/jobs",
            json={
                "query_image_id": query["id"],
                "target_video_id": video["id"],
                "parameters": {
                    "sample_fps": 2.0,
                    "window_sec": 2.0,
                    "stride_sec": 0.5,
                    "top_frame_ratio": 0.5,
                    "nms_iou": 0.3,
                },
            },
        )
        created.raise_for_status()
        job_id = created.json()["id"]

    worker = Worker()
    if not worker.run_once():
        raise RuntimeError("The queued smoke-test job was not claimed")

    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        print(json.dumps(job, ensure_ascii=False, indent=2))
        if job["status"] != "succeeded" or job["result"] is None:
            raise RuntimeError("Smoke-test job did not succeed")


if __name__ == "__main__":
    main()
