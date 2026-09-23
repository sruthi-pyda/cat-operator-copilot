"""Webcam registration and login for the three real demo operators.

    python -m features.passport.registration register --operator-id OP1003
    python -m features.passport.registration login

Registration captures a burst of frames, keeps the ones with a detectable face,
and stores an embedding per frame. Frames are written alongside the embeddings
under `data/face_registrations/` (gitignored) so nothing leaves the laptop, and
only the embeddings are ever read back.

Only OP1001-OP1003 are real people; every other operator in the dataset is
synthetic and is never registered.

stdout is switched to UTF-8 because DeepFace's logger prints emoji and the
default Windows console encoding raises UnicodeEncodeError mid-import (D014).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

from features.passport.biometric import (
    DeepFaceRecognizer,
    EmbeddingStore,
    NoFaceDetectedError,
    identify,
    register_operator,
)
from shared.config import load_settings

DEFAULT_FRAME_COUNT = 5
DEFAULT_FRAME_DELAY_SEC = 0.8
WARMUP_FRAMES = 10  # webcams need a few reads before exposure settles


def _use_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


class CameraError(RuntimeError):
    pass


def _build_recognizer(settings: dict) -> DeepFaceRecognizer:
    biometric = settings["biometric"]
    return DeepFaceRecognizer(
        model_name=biometric["face_model"],
        detector_backend=biometric.get("face_detector_backend", "retinaface"),
    )


def capture_frames(
    count: int = DEFAULT_FRAME_COUNT,
    camera_index: int = 0,
    delay_sec: float = DEFAULT_FRAME_DELAY_SEC,
    show_preview: bool = True,
    countdown_sec: int = 0,
) -> list[Any]:
    """Grab `count` frames from the webcam, with a live preview window.

    `countdown_sec` gives the person time to get in front of the camera; the
    warm-up reads run during it so the preview is live while they position.
    """
    import cv2

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise CameraError(
            f"Could not open camera {camera_index}. Close any app using the webcam and retry."
        )

    frames = []
    try:
        for _ in range(WARMUP_FRAMES):
            camera.read()

        for remaining in range(countdown_sec, 0, -1):
            print(f"  starting in {remaining}...")
            ok, frame = camera.read()
            if ok and show_preview:
                cv2.imshow("Registration - look at the camera", frame)
                cv2.waitKey(1)
            time.sleep(1)

        for index in range(count):
            time.sleep(delay_sec)
            ok, frame = camera.read()
            if not ok:
                raise CameraError(f"Camera stopped returning frames after {len(frames)} captures.")
            frames.append(frame)
            print(f"  captured frame {index + 1}/{count}")
            if show_preview:
                cv2.imshow("Registration - look at the camera", frame)
                cv2.waitKey(1)
    finally:
        camera.release()
        if show_preview:
            cv2.destroyAllWindows()
    return frames


def save_frames(frames: list[Any], operator_id: str, store: EmbeddingStore) -> list[Path]:
    """Keep the captures locally next to the embeddings; never uploaded."""
    import cv2

    folder = store.root / operator_id / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = folder / f"frame_{index:02d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def run_register(
    operator_id: str,
    frame_count: int,
    camera_index: int,
    show_preview: bool,
    countdown_sec: int = 5,
    settings: Optional[dict] = None,
) -> int:
    settings = settings or load_settings()
    store = EmbeddingStore.from_settings(settings)
    recognizer = _build_recognizer(settings)

    print(f"Registering {operator_id} with {frame_count} frames. Look at the camera.")
    frames = capture_frames(
        frame_count, camera_index, show_preview=show_preview, countdown_sec=countdown_sec
    )
    save_frames(frames, operator_id, store)

    try:
        result = register_operator(operator_id, frames, recognizer, store)
    except NoFaceDetectedError as exc:
        print(f"Registration failed: {exc}")
        print("Try again with more light and your face filling more of the frame.")
        return 1

    print(
        f"Registered {operator_id}: {result.embeddings_stored} embeddings stored, "
        f"{result.frames_rejected} frames had no detectable face."
    )
    print(f"Stored at {result.path}")
    return 0


def run_login(camera_index: int, show_preview: bool, settings: Optional[dict] = None) -> int:
    settings = settings or load_settings()
    store = EmbeddingStore.from_settings(settings)
    registered = store.registered_operator_ids()
    if not registered:
        print("No operators are registered yet. Run the register command first.")
        return 1

    recognizer = _build_recognizer(settings)
    print(f"Registered operators: {', '.join(registered)}")
    print("Capturing one frame for login...")

    frames = capture_frames(1, camera_index, show_preview=show_preview)
    try:
        result = identify(frames[0], recognizer, store, settings=settings)
    except NoFaceDetectedError:
        print("No face detected in the captured frame. Not authenticated.")
        return 1

    print(f"\n{result.to_dict()}")
    if result.authenticated:
        print(f"\nAuthenticated as {result.operator_id} (confidence {result.confidence:.4f}).")
        print("Authorization for a specific machine is checked separately by the Passport.")
        return 0

    print(
        f"\nNot authenticated. Best score {result.confidence:.4f} is below the "
        f"threshold {result.threshold}. No session will be created."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operator Passport face registration and login.")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="Register one operator from the webcam.")
    register.add_argument("--operator-id", required=True, help="e.g. OP1003")
    register.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT)
    register.add_argument(
        "--countdown", type=int, default=5, help="Seconds before the first capture."
    )

    sub.add_parser("login", help="Capture one frame and identify the operator.")

    for name in ("register", "login"):
        sub.choices[name].add_argument("--camera", type=int, default=0)
        sub.choices[name].add_argument("--no-preview", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _use_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.command == "register":
        return run_register(
            args.operator_id,
            args.frames,
            args.camera,
            show_preview=not args.no_preview,
            countdown_sec=args.countdown,
        )
    return run_login(args.camera, show_preview=not args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
