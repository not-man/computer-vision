"""GPU-only candidate perception. Pilot first, then full cameras with one continuous tracker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT

from mct.embedder import create_embedder
from mct.enhanced import infer
from mct.perception import CONFIGS, PROCESS_SIZE, TRACKER, encode_mask, touches_edge
from mct.video import read_frames

COLUMNS = ["camera", "frame", "t_s", "raw_track", "conf", "x1", "y1", "x2", "y2", "touches_edge", "mask_area", "mask_rle"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIGS / "enhanced.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--pilot", action="store_true", help="30-second crowd and far-zone samples; never use as full results")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.device == "cpu":
        parser.error("GPU required: run in Google Colab with GPU")
    baseline, out = args.baseline.resolve(), args.out.resolve()
    if baseline == out or baseline in out.parents or out in baseline.parents:
        parser.error("Baseline and candidate must be separate directories")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    clocks = yaml.safe_load((CONFIGS / "clocks.yaml").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        (out / "PILOT_ONLY").write_text("Partial camera samples, not a full perception dataset", encoding="utf-8")
    elif (out / "PILOT_ONLY").exists():
        parser.error("Use a different output directory for the full run")
    embedder = create_embedder(args.device, cfg["reid_model"])
    if embedder.name != "boxmot:" + cfg["reid_model"]:
        raise RuntimeError(f"Expected OSNet, got {embedder.name}; refusing weaker fallback")
    model = YOLO(cfg["model"])
    model_hash = sha(model.ckpt_path)
    code_root = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(code_root)): sha(p) for p in sorted((code_root / "src/mct").glob("*.py"))}
    code["runner"] = sha(__file__)
    for camera, clock in clocks.items():
        if camera not in cfg["cameras"]:
            if not args.pilot:
                for suffix in (".parquet", "_emb.npz", "_summary.json"):
                    shutil.copyfile(baseline / (camera + suffix), out / (camera + suffix))
            continue
        video = args.videos / (camera + ".mkv")
        signature = hashlib.sha256(json.dumps({"cfg": cfg, "clocks": clocks, "code": code, "video": sha(video),
                                              "weights": model_hash, "pilot": args.pilot}, sort_keys=True).encode()).hexdigest()
        summary_path = out / (camera + "_summary.json")
        if summary_path.exists():
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            if old.get("signature") != signature:
                raise ValueError(f"Changed inputs for {camera}; choose a new --out directory")
            if all((out / name).exists() and sha(out / name) == value for name, value in old["files"].items()):
                print(camera, "verified; skipping", flush=True)
                continue
        tracker = BOTSORT(SimpleNamespace(**yaml.safe_load(Path(TRACKER).read_text(encoding="utf-8"))))
        rows, keys, vectors, provenance = [], [], [], []
        last = {}
        start = max(0, 2720 - clock["clock_at_frame_0"]) if args.pilot else 0
        started = time.perf_counter()
        count = 0
        for frame_index, native in read_frames(video, tuple(cfg["native_size"]), clock["fps"], cfg["fps"], start_s=start,
                                               duration_s=30 if args.pilot else None):
            count += 1
            t = clock["clock_at_frame_0"] + frame_index / clock["fps"]
            base, detections = infer(model, native, camera, t, cfg, args.device)
            data = np.array([[*d.box, d.score, 0] for d in detections], dtype=np.float32).reshape(-1, 6)
            tracked = tracker.update(Boxes(data, base.shape[:2]).numpy(), base)
            to_embed = []
            dense = cfg.get("dense_reid", {})
            interval = dense.get("interval_s", .25) if dense.get("enabled") and any(a <= t < b for a, b in dense.get("windows", [])) else 1.0
            for track in tracked:
                x1, y1, x2, y2, raw, score, cls, detection_index = track
                raw = int(raw)
                item = detections[int(detection_index)]
                box = np.array([x1, y1, x2, y2])
                edge = touches_edge(box, PROCESS_SIZE)
                rows.append(dict(zip(COLUMNS, [camera, frame_index, round(t, 3), raw, float(score), *map(float, box), edge,
                                                int(item.mask.sum()), encode_mask(item.mask)])))
                provenance.append({"frame": frame_index, "raw_track": raw, "origin": item.origin, "clipped": item.clipped})
                # Keep mixed crops out of the appearance gallery.
                intersection = [max(0, min(x2, d.box[2])-max(x1, d.box[0])) * max(0, min(y2, d.box[3])-max(y1, d.box[1]))
                                / max((x2-x1)*(y2-y1), 1) for d in detections if d is not item]
                if not edge and not item.clipped and max(intersection, default=0) < .5 and t-last.get(raw, -1e9) >= interval:
                    last[raw] = t
                    to_embed.append((raw, box))
            if to_embed:
                scale = np.array([native.shape[1]/PROCESS_SIZE[0], native.shape[0]/PROCESS_SIZE[1]] * 2)
                vectors.append(embedder.embed(native, np.array([b for _, b in to_embed]) * scale))
                keys.extend((frame_index, raw) for raw, _ in to_embed)
            if count % 100 == 0:
                print(f"{camera}: {count} frames, {time.perf_counter()-started:.1f}s", flush=True)
            if args.pilot and count % 50 == 0:
                preview = base.copy()
                for tr in tracked:
                    box = tr[:4].astype(int)
                    source = detections[int(tr[-1])].origin
                    color = (0, 230, 80) if source != "full" else (255, 180, 0)
                    cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), color, 2)
                    cv2.putText(preview, f"{int(tr[4])} {source}", tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                cv2.imwrite(str(out / f"{camera}_preview_{frame_index}.jpg"), preview)
        if count == 0 or (not args.pilot and count < clock["frames"] / clock["fps"] * cfg["fps"] - 2):
            raise RuntimeError(f"Truncated video: {camera}, {count} sampled frames")
        # Completion marker is published last. A crash leaves no apparently finished camera.
        files = {}
        for suffix, write in [
            (".parquet", lambda p: pd.DataFrame(rows, columns=COLUMNS).to_parquet(p, index=False)),
            ("_emb.npz", lambda p: np.savez_compressed(p, keys=np.array(keys, np.int64).reshape(-1, 2), vectors=np.vstack(vectors) if vectors else np.empty((0, 512), np.float32))),
            ("_sources.parquet", lambda p: pd.DataFrame(provenance).to_parquet(p, index=False)),
        ]:
            path = out / (camera + suffix)
            # np.savez appends .npz unless the temporary name ends with .npz.
            temp = path.with_name("pending_" + path.name)
            write(temp)
            os.replace(temp, path)
            files[path.name] = sha(path)
        summary = {"signature": signature, "camera": camera, "model": cfg["model"], "embedder": embedder.name,
                   "fps": cfg["fps"], "imgsz": cfg["imgsz"], "conf": cfg["conf"], "frames": count,
                   "observations": len(rows), "seconds": time.perf_counter()-started, "pilot": args.pilot, "files": files}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
