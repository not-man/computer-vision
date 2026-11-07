"""Собирает ролик-витрину из готовых минутных частей полной записи."""
import argparse
import json
from pathlib import Path

import cv2

from mct.showcase import build_showcase, pick_moments
from mct.video import clock_text


def save_overview(video: Path, output: Path, seconds: float) -> None:
    """Кадр для README берётся из того же ролика, что и готовая витрина."""
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, min(6.0, seconds / 2) * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Не удалось получить обзорный кадр из {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Не удалось сохранить обзорный кадр в {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks/tracks.json"))
    parser.add_argument("--render", type=Path, default=Path("render/full"))
    parser.add_argument("--overview", type=Path, default=Path("artifacts/demo/overview.jpg"))
    args = parser.parse_args()

    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))["tracks"]
    manifest = json.loads((args.render / "manifest.json").read_text(encoding="utf-8"))
    moments = pick_moments(tracks)
    for moment in moments:
        spans = ", ".join(f"{clock_text(start)[:8]}–{clock_text(end)[:8]}" for start, end in moment.spans)
        print(f"{moment.title} [{spans}]: {moment.note}")
    seconds = build_showcase(moments, manifest, args.render, args.render / "showcase.mp4")
    save_overview(args.render / "showcase.mp4", args.overview, seconds)
    print(f"Витрина: {seconds:.1f} с, {(args.render / 'showcase.mp4').stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
