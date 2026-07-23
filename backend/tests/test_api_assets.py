from __future__ import annotations

from io import BytesIO

from PIL import Image


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (220, 80, 40)).save(output, "PNG")
    return output.getvalue()


def test_upload_list_and_range_content(client) -> None:
    response = client.post(
        "/api/assets/upload",
        files={"file": ("query.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["kind"] == "image"
    assert asset["thumbnail_url"]

    listing = client.get("/api/assets", params={"kind": "image"})
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1

    partial = client.get(
        asset["content_url"],
        headers={"Range": "bytes=0-9"},
    )
    assert partial.status_code == 206
    assert len(partial.content) == 10
    assert partial.headers["accept-ranges"] == "bytes"


def test_scan_is_idempotent(client, settings) -> None:
    image_path = settings.scan_roots["test"] / "sample.png"
    image_path.write_bytes(png_bytes())

    first = client.post(
        "/api/assets/scan",
        json={"scan_root_id": "test", "recursive": True},
    )
    second = client.post(
        "/api/assets/scan",
        json={"scan_root_id": "test", "recursive": True},
    )
    assert first.status_code == 200
    assert first.json()["created"] == 1
    assert second.status_code == 200
    assert second.json()["created"] == 0
    assert second.json()["skipped"] == 1


def test_scan_rejects_path_escape(client) -> None:
    response = client.post(
        "/api/assets/scan",
        json={
            "scan_root_id": "test",
            "relative_dir": "../outside",
            "recursive": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSAFE_SCAN_PATH"


def test_invalid_range_uses_stable_error_shape(client) -> None:
    created = client.post(
        "/api/assets/upload",
        files={"file": ("query.png", png_bytes(), "image/png")},
    ).json()
    response = client.get(
        created["content_url"],
        headers={"Range": "bytes=999999-1000000"},
    )
    assert response.status_code == 416
    assert response.json()["error"]["code"] == "INVALID_RANGE"
