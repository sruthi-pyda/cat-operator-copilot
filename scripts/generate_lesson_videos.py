"""Render each micro-lesson to a short tutorial video.

    python scripts/generate_lesson_videos.py

Frames are drawn with Pillow and encoded with OpenCV, so nothing is downloaded
and no external service is involved. The content is the lesson's own text from
features/training/content/lessons.yaml -- the video cannot say anything the
lesson does not, which is the same grounding rule the Buddy follows.

Output goes to features/training/content/media/<lesson_id>.mp4 (gitignored, so
each machine renders its own). Set `video_url: "L001.mp4"` on the lesson to use
one.

Styling matches the Command Deck tokens in app/ui/dashboard.py.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.training.lessons import Lesson, load_content  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 24
MEDIA_DIR = PROJECT_ROOT / "features" / "training" / "content" / "media"

BG = (28, 26, 23)           # BGR of #171A1C
PANEL = (40, 36, 32)        # #202428
TEXT = (233, 237, 237)      # #EDEDE9
MUTED = (166, 160, 154)     # #9AA0A6
ACCENT = (68, 181, 242)     # #F2B544
OK = (143, 168, 127)        # #7FA88F

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


def _slide(title: str, lines: Sequence[str], footer: str, kicker: str = "") -> Image.Image:
    """One frame: kicker, title, wrapped body lines, footer rule."""
    image = Image.new("RGB", (WIDTH, HEIGHT), _rgb(BG))
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, 6, HEIGHT], fill=_rgb(ACCENT))          # left accent rule
    draw.rectangle([70, 70, WIDTH - 70, HEIGHT - 90], fill=_rgb(PANEL))

    y = 120
    if kicker:
        draw.text((110, y), kicker.upper(), font=_font(20), fill=_rgb(ACCENT))
        y += 42

    for line in textwrap.wrap(title, width=42):
        draw.text((110, y), line, font=_font(46, bold=True), fill=_rgb(TEXT))
        y += 58
    y += 18

    for line in lines:
        bullet = line.startswith("• ")
        body = line[2:] if bullet else line
        for i, wrapped in enumerate(textwrap.wrap(body, width=68)):
            if bullet and i == 0:
                draw.ellipse([112, y + 13, 120, y + 21], fill=_rgb(ACCENT))
            draw.text((140 if bullet else 110, y), wrapped,
                      font=_font(28), fill=_rgb(TEXT if bullet else MUTED))
            y += 40
        y += 10

    draw.text((110, HEIGHT - 128), footer, font=_font(20), fill=_rgb(MUTED))
    return image


def _with_progress(image: Image.Image, fraction: float) -> np.ndarray:
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, HEIGHT - 6), (WIDTH, HEIGHT), PANEL, -1)
    cv2.rectangle(frame, (0, HEIGHT - 6), (int(WIDTH * fraction), HEIGHT), ACCENT, -1)
    return frame


def _slides_for(lesson: Lesson) -> list[tuple[str, list[str], str]]:
    footer = f"{lesson.lesson_id} · synthetic training content · not official Caterpillar material"
    slides: list[tuple[str, list[str], str]] = [
        (lesson.title, [lesson.summary], footer),
    ]
    if lesson.objectives:
        slides.append(("What you'll get out of it",
                       [f"• {o}" for o in lesson.objectives], footer))
    for index, point in enumerate(lesson.key_points, start=1):
        slides.append((f"Key point {index} of {len(lesson.key_points)}",
                       [f"• {point}"], footer))
    if lesson.practice_prompt:
        slides.append(("Try this on your next shift", [lesson.practice_prompt], footer))
    slides.append(("Now take the quiz",
                   ["Passing the quiz does not close the issue.",
                    "Only a measured improvement does."], footer))
    return slides


# Container per codec. Browsers cannot decode mp4v -- Chrome reports
# canPlayType('video/mp4; codecs="mp4v.20.8"') as "" and fails the file with
# DEMUXER_ERROR_NO_SUPPORTED_STREAMS. H.264 needs the OpenH264 DLL, which is not
# present. VP8 in WebM is decodable everywhere and needs nothing extra.
CONTAINER = {"VP80": ".webm", "VP90": ".webm"}


def render_lesson(lesson: Lesson, seconds_per_slide: float, fourcc: str) -> Path:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIA_DIR / f"{lesson.lesson_id}{CONTAINER.get(fourcc, '.mp4')}"
    slides = _slides_for(lesson)
    per_slide = max(1, int(seconds_per_slide * FPS))
    total = per_slide * len(slides)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for fourcc {fourcc!r}")

    written = 0
    for title, lines, footer in slides:
        image = _slide(title, lines, footer, kicker=lesson.title if title != lesson.title else "")
        for _ in range(per_slide):
            writer.write(_with_progress(image, written / total))
            written += 1
    writer.release()
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render micro-lessons to tutorial videos.")
    parser.add_argument("--lesson", help="Only this lesson id (e.g. L003)")
    parser.add_argument("--seconds", type=float, default=4.0, help="Seconds per slide")
    parser.add_argument("--fourcc", default="VP80",
                        help="OpenCV fourcc. VP80 (WebM) is the default because browsers "
                             "cannot decode mp4v and H.264 needs a DLL that is not bundled.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    lessons = load_content().lessons
    if args.lesson:
        lessons = tuple(l for l in lessons if l.lesson_id == args.lesson)
        if not lessons:
            print(f"No lesson {args.lesson}")
            return 1

    for lesson in lessons:
        path = render_lesson(lesson, args.seconds, args.fourcc)
        size_kb = path.stat().st_size / 1024
        print(f"{lesson.lesson_id}  {path.name:<12} {size_kb:8.0f} KB  {lesson.title}")

    suffix = CONTAINER.get(args.fourcc, ".mp4")
    print(f"\nSet video_url on a lesson in lessons.yaml, e.g. video_url: \"L001{suffix}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
