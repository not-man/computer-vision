"""Все треки снимками из исходных кадров, по странице на 16 треков: так видно, один ли человек в треке. Чтение видео занимает несколько минут."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mct.render import BACKGROUND, Stamps, paste
from mct.video import clock_text, read_selected_frames

SOURCE_SIZE = (2592, 1944)
SCALE = 2  # исходный кадр вдвое крупнее рабочего, в котором записаны рамки
HEIGHT, PER_TRACK, PER_PAGE, WIDTH = 150, 15, 16, 1500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--videos", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/test-results/audit/source_res"))
    parser.add_argument("--only", type=int, nargs="+", default=None, help="только эти треки: например, изменившиеся после правки")
    args = parser.parse_args()

    tracks = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    if args.only:
        tracks = [track for track in tracks if track["track_id"] in args.only]
    table = pd.read_parquet(args.tracks / "observations.parquet", columns=["tracklet", "camera", "frame", "t_s", "x1", "y1", "x2", "y2"])
    chosen, wanted = {}, {}
    for track in tracks:
        rows = table[table.tracklet.isin(track["tracklets"])].sort_values("t_s")
        picks = rows.iloc[sorted(set(np.linspace(0, len(rows) - 1, min(PER_TRACK, len(rows))).round().astype(int).tolist()))]
        chosen[track["track_id"]] = picks
        for row in picks.itertuples():
            wanted.setdefault(row.camera, []).append(row)
    crops = {}

    def read(camera: str) -> None:
        by_frame: dict[int, list] = {}
        for row in wanted[camera]:
            by_frame.setdefault(row.frame, []).append(row)
        for number, image in read_selected_frames(args.videos / f"{camera}.mkv", by_frame, SOURCE_SIZE):
            for row in by_frame[number]:
                crop = image[int(row.y1 * SCALE):int(row.y2 * SCALE), int(row.x1 * SCALE):int(row.x2 * SCALE)]
                if crop.size:
                    size = (max(int(crop.shape[1] * HEIGHT / crop.shape[0]), 1), HEIGHT)
                    crops[(camera, number, row.Index)] = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if crop.shape[0] > HEIGHT else cv2.INTER_CUBIC)[:, :130].copy()

    with ThreadPoolExecutor(len(wanted)) as pool:
        list(pool.map(read, wanted))
    args.out.mkdir(parents=True, exist_ok=True)
    stamps = Stamps()
    for page, start in enumerate(range(0, len(tracks), PER_PAGE), 1):
        part = tracks[start:start + PER_PAGE]
        sheet = np.full(((HEIGHT + 36) * len(part), WIDTH, 3), BACKGROUND, np.uint8)
        for n, track in enumerate(part):
            top = n * (HEIGHT + 36)
            seconds = (track["t_end_ms"] - track["t_start_ms"]) / 1000
            paste(sheet, stamps.text(f"#{track['track_id']}  {track['start'][3:8]}–{track['end'][3:8]}  {seconds:.0f} с  {track['close_zone']}", 12, bold=True), 4, top + 1)
            x = 4
            for row in chosen[track["track_id"]].itertuples():
                crop = crops.get((row.camera, row.frame, row.Index))
                if crop is None or x + crop.shape[1] > WIDTH - 4:
                    continue
                sheet[top + 18:top + 18 + HEIGHT, x:x + crop.shape[1]] = crop
                paste(sheet, stamps.text(f"{row.camera[-1]} {clock_text(row.t_s)[3:8]}", 9), x, top + 18 + HEIGHT + 1)
                x += crop.shape[1] + 3
        cv2.imwrite(str(args.out / f"strips_{page}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        print(page, sheet.shape, flush=True)


if __name__ == "__main__":
    main()
