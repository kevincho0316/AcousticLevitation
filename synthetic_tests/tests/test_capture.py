from __future__ import annotations

from capture.capture import _resolve_device_indices


def test_resolve_device_indices_falls_back_to_detected_order(monkeypatch):
    monkeypatch.setattr("capture.capture._probe_camera_indices", lambda max_index=16: [2, 4, 6])

    cameras = [
        {"id": "cam_1", "serial": "ABC123"},
        {"id": "cam_2", "serial": "DEF456"},
        {"id": "cam_3", "serial": "GHI789"},
    ]

    resolved = _resolve_device_indices(cameras, serial_to_index={})

    assert resolved == [2, 4, 6]


def test_resolve_device_indices_prefers_explicit_serial_and_device_index(monkeypatch):
    monkeypatch.setattr("capture.capture._probe_camera_indices", lambda max_index=16: [1, 3, 5, 7])

    cameras = [
        {"id": "cam_1", "serial": "ABC123"},
        {"id": "cam_2", "serial": "DEF456", "device_index": 9},
        {"id": "cam_3", "serial": "GHI789"},
    ]

    resolved = _resolve_device_indices(
        cameras,
        serial_to_index={"ABC123": 5, "GHI789": 7},
    )

    assert resolved == [5, 9, 7]
