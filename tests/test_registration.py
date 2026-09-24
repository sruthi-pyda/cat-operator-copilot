"""Tests for the webcam registration CLI (Feature 01).

`cv2` is replaced with a fake module so these run without a camera. The capture
loop, the local-only storage rule and the argument surface are all testable;
only the actual image acquisition is not, and that is the part the fake stands
in for.
"""
from __future__ import annotations

import sys
import types

import pytest

from features.passport import registration
from features.passport.biometric import EmbeddingStore
from features.passport.registration import (
    CameraError,
    build_parser,
    capture_frames,
    save_frames,
)


class FakeCamera:
    def __init__(self, opened=True, fail_at=None):
        self._opened = opened
        self._fail_at = fail_at
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        self.reads += 1
        if self._fail_at is not None and self.reads > self._fail_at:
            return False, None
        return True, f"frame_{self.reads}"

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    camera = FakeCamera()
    written = []

    module = types.SimpleNamespace(
        VideoCapture=lambda index: camera,
        imshow=lambda *a, **k: None,
        waitKey=lambda *a, **k: -1,
        destroyAllWindows=lambda: None,
        imwrite=lambda path, frame: written.append((path, frame)) or True,
    )
    module.camera = camera
    module.written = written
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


def test_capture_returns_the_requested_frame_count(fake_cv2):
    frames = capture_frames(count=3, delay_sec=0, show_preview=False)
    assert len(frames) == 3
    assert fake_cv2.camera.released is True


def test_warmup_frames_are_discarded_not_registered(fake_cv2):
    """Early webcam reads are dark; they must not become embeddings."""
    frames = capture_frames(count=2, delay_sec=0, show_preview=False)
    assert len(frames) == 2
    assert fake_cv2.camera.reads == registration.WARMUP_FRAMES + 2


def test_an_unopenable_camera_raises_an_actionable_error(fake_cv2, monkeypatch):
    monkeypatch.setattr(fake_cv2, "VideoCapture", lambda index: FakeCamera(opened=False))
    with pytest.raises(CameraError, match="Could not open camera"):
        capture_frames(count=1, delay_sec=0, show_preview=False)


def test_camera_failing_mid_capture_raises_and_still_releases(fake_cv2):
    fake_cv2.camera._fail_at = registration.WARMUP_FRAMES + 1
    with pytest.raises(CameraError, match="stopped returning frames"):
        capture_frames(count=3, delay_sec=0, show_preview=False)
    assert fake_cv2.camera.released is True


def test_frames_are_saved_under_the_operator_folder_only(fake_cv2, tmp_path):
    store = EmbeddingStore(tmp_path / "face_registrations")
    paths = save_frames(["f1", "f2"], "OP1003", store)
    assert len(paths) == 2
    assert all(p.parent == store.root / "OP1003" / "frames" for p in paths)
    assert all(str(p).endswith(".jpg") for p in paths)
    assert len(fake_cv2.written) == 2


# --- CLI surface -------------------------------------------------------------

def test_register_command_parses():
    args = build_parser().parse_args(["register", "--operator-id", "OP1003", "--frames", "7"])
    assert (args.command, args.operator_id, args.frames, args.camera) == (
        "register", "OP1003", 7, 0,
    )


def test_login_command_parses_with_preview_toggle():
    args = build_parser().parse_args(["login", "--no-preview", "--camera", "1"])
    assert args.command == "login"
    assert args.no_preview is True
    assert args.camera == 1


def test_register_requires_an_operator_id():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["register"])


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_login_reports_when_nobody_is_registered(tmp_path, capsys):
    settings = {
        "biometric": {
            "face_confidence_threshold": 0.70,
            "face_model": "ArcFace",
            "registration_image_dir": str(tmp_path / "empty"),
        }
    }
    assert registration.run_login(camera_index=0, show_preview=False, settings=settings) == 1
    assert "No operators are registered" in capsys.readouterr().out
