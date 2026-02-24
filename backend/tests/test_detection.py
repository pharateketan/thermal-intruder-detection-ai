"""
Tests for the ThermalEye detection API.
Run with:  pytest tests/ -v
"""

import io
import json
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.model_manager import ModelManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dummy_image(width=640, height=512, mode="RGB") -> bytes:
    """Create a small in-memory PNG to use as a test upload."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def mock_yolo_result(num_boxes=2):
    """Return a fake ultralytics Results object."""
    import torch

    result = MagicMock()
    result.boxes = MagicMock()
    result.boxes.xyxy = torch.tensor(
        [[50, 60, 150, 200], [300, 100, 420, 280]], dtype=torch.float32
    )[:num_boxes]
    result.boxes.conf = torch.tensor([0.87, 0.72])[:num_boxes]
    result.boxes.cls = torch.tensor([0, 1], dtype=torch.float32)[:num_boxes]
    return [result]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_model():
    """Patch ModelManager so tests never need a real .pt file."""
    model_mock = MagicMock()
    model_mock.predict.return_value = mock_yolo_result(2)

    with patch.object(ModelManager, "_model", model_mock), \
         patch.object(ModelManager, "is_loaded", True):
        yield model_mock


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "model_loading")
    assert "model_loaded" in body


# ── Detection ─────────────────────────────────────────────────────────────────

def test_detect_returns_200(client):
    img_bytes = make_dummy_image()
    r = client.post(
        "/api/detect",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200


def test_detect_response_schema(client):
    img_bytes = make_dummy_image()
    r = client.post(
        "/api/detect",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    body = r.json()
    assert body["success"] is True
    assert "detections" in body
    assert "summary" in body
    assert "inference_ms" in body
    assert body["image_width"] == 640
    assert body["image_height"] == 512


def test_detect_counts_classes(client):
    img_bytes = make_dummy_image()
    r = client.post("/api/detect", files={"file": ("t.png", img_bytes, "image/png")})
    s = r.json()["summary"]
    assert s["total"] == 2
    assert s["persons"] == 1
    assert s["vehicles"] == 1
    assert s["threat_detected"] is True


def test_detect_bbox_fields(client):
    img_bytes = make_dummy_image()
    r = client.post("/api/detect", files={"file": ("t.png", img_bytes, "image/png")})
    det = r.json()["detections"][0]
    bbox = det["bbox"]
    for key in ("x1", "y1", "x2", "y2", "width", "height", "x1_norm", "y1_norm", "x2_norm", "y2_norm"):
        assert key in bbox, f"Missing bbox field: {key}"


def test_detect_custom_conf(client):
    img_bytes = make_dummy_image()
    r = client.post(
        "/api/detect?conf=0.8&iou=0.4",
        files={"file": ("t.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["conf_threshold"] == 0.8


def test_detect_invalid_file_type(client):
    r = client.post(
        "/api/detect",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 415


def test_detect_no_file(client):
    r = client.post("/api/detect")
    assert r.status_code == 422   # FastAPI validation error


def test_detect_conf_out_of_range(client):
    img_bytes = make_dummy_image()
    r = client.post(
        "/api/detect?conf=1.5",
        files={"file": ("t.png", img_bytes, "image/png")},
    )
    assert r.status_code == 422
