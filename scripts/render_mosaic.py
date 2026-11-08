"""Мозаика всех камер с силуэтами, живым планом и плашками: минутные части H.264 и манифест для интерфейса."""
import argparse
import json
import time
from pathlib import Path

from mct.render import render_mosaic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("data"))
    parser.add_argument("--plan-image", type=Path, default=Path("data/office.jpg"))
    parser.add_argument("--out", type=Path, default=Path("render"))
    parser.add_argument("--clock-start", type=float, default=None, help="секунды от 17:00:00; по умолчанию вся запись")
    parser.add_argument("--clock-end", type=float, default=None)
    parser.add_argument("--chunk-s", type=int, default=60)
    parser.add_argument("--hide-fragments", action="store_true", help="не рисовать контуры обломков без итогового трека")
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = render_mosaic(
        args.videos, args.perception, args.tracks, args.plan_image, args.out, args.clock_start, args.clock_end,
        args.chunk_s, not args.hide_fragments,
    )
    seconds = time.perf_counter() - started
    frames = (manifest["clock_end_s"] - manifest["clock_start_s"]) * manifest["fps"]
    rendering = sum(chunk["render_s"] for chunk in manifest["chunks"])
    print(json.dumps({
        "частей": len(manifest["chunks"]), "МБ": [chunk["mb"] for chunk in manifest["chunks"]][:6],
        "кадров": int(frames), "секунд_всего": round(seconds, 1), "перемотка_с": round(seconds - rendering, 1),
        "кадр/с_рендера": round(frames / max(rendering, 0.1), 1),
        "событий": len(manifest["events"]), "треков": manifest["total_tracks"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
