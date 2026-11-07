"""Собирает ролик-витрину из готовых минутных частей полной записи."""
import argparse
import json
from pathlib import Path

from mct.showcase import build_showcase, pick_moments
from mct.video import clock_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks/tracks.json"))
    parser.add_argument("--render", type=Path, default=Path("render/full"))
    args = parser.parse_args()

    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))["tracks"]
    manifest = json.loads((args.render / "manifest.json").read_text(encoding="utf-8"))
    moments = pick_moments(tracks)
    for moment in moments:
        spans = ", ".join(f"{clock_text(start)[:8]}–{clock_text(end)[:8]}" for start, end in moment.spans)
        print(f"{moment.title} [{spans}]: {moment.note}")
    seconds = build_showcase(moments, manifest, args.render, args.render / "showcase.mp4")
    print(f"Витрина: {seconds:.1f} с, {(args.render / 'showcase.mp4').stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
