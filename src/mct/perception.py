"""Тяжёлая часть конвейера: видео → наблюдения, маски и векторы ReID. Запускается на GPU один раз."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_utils
from ultralytics import YOLO

from mct.embedder import Embedder
from mct.video import read_frames

PROCESS_SIZE = (1296, 972)  # половина исходных 2592×1944, этого хватает при таком размере людей
EDGE_MARGIN_PX = 4
CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TRACKER = str(CONFIGS / "botsort.yaml")


def encode_mask(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def touches_edge(box: np.ndarray, size: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = box
    width, height = size
    return x1 < EDGE_MARGIN_PX or y1 < EDGE_MARGIN_PX or x2 > width - EDGE_MARGIN_PX or y2 > height - EDGE_MARGIN_PX


def process_camera(
    video: Path,
    out_dir: Path,
    embedder: Embedder,
    model_name: str,
    device: str,
    fps: float = 10.0,
    imgsz: int = 960,
    conf: float = 0.1,
    embed_every_s: float = 1.0,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict:
    """Пишет <камера>.parquet и <камера>_emb.npz; файл summary появляется последним и означает готовность."""
    camera = video.stem
    clock = yaml.safe_load((CONFIGS / "clocks.yaml").read_text())[camera]
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)  # новая модель на камеру, чтобы состояние трекера не переходило между ними
    rows, emb_rows, emb_vectors = [], [], []
    last_embedded: dict[int, float] = {}
    started = time.perf_counter()
    frames = 0
    for index, frame in read_frames(video, PROCESS_SIZE, clock["fps"], fps, start_s, duration_s):
        t_s = clock["clock_at_frame_0"] + index / clock["fps"]  # общее время всех камер
        frames += 1
        if frames % 1000 == 0:
            print(f"{camera}: {frames} кадров, {time.perf_counter() - started:.0f} с", flush=True)
        result = model.track(
            frame, imgsz=imgsz, conf=conf, classes=[0], persist=True, tracker=TRACKER,
            retina_masks=True, device=device, verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None or result.masks is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        masks = result.masks.data.cpu().numpy() > 0.5

        to_embed = []
        for box, score, track_id, mask in zip(boxes, scores, track_ids, masks):
            edge = touches_edge(box, PROCESS_SIZE)
            rows.append({
                "camera": camera, "frame": index, "t_s": round(t_s, 3), "raw_track": int(track_id),
                "conf": float(score), "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
                "touches_edge": edge, "mask_area": int(mask.sum()), "mask_rle": encode_mask(mask),
            })
            if not edge and t_s - last_embedded.get(int(track_id), -1e9) >= embed_every_s:
                last_embedded[int(track_id)] = t_s
                to_embed.append((int(track_id), box))
        if to_embed:
            vectors = embedder.embed(frame, np.array([box for _, box in to_embed]))
            emb_rows += [(index, track_id) for track_id, _ in to_embed]
            emb_vectors.append(vectors)

    frame_table = pd.DataFrame(rows)
    frame_table.to_parquet(out_dir / f"{camera}.parquet", index=False)
    np.savez_compressed(
        out_dir / f"{camera}_emb.npz",
        keys=np.array(emb_rows, np.int64).reshape(-1, 2),
        vectors=np.vstack(emb_vectors) if emb_vectors else np.empty((0, 0), np.float32),
    )
    summary = {
        "camera": camera, "model": model_name, "imgsz": imgsz, "fps": fps, "conf": conf,
        "embedder": embedder.name, "frames": frames, "observations": len(rows),
        "raw_tracks": int(frame_table["raw_track"].nunique()) if rows else 0,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / f"{camera}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
