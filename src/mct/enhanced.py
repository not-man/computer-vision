"""Optional native-resolution ROI inference; the baseline perception path is untouched."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: np.ndarray
    score: float
    mask: np.ndarray
    origin: str
    clipped: bool = False


def project_detection(box, score, mask, region, native_size, output_size, origin):
    x, y, width, height = region
    nw, nh = native_size
    ow, oh = output_size
    sx, sy = ow / nw, oh / nh
    mapped = np.asarray(box, dtype=float) + [x, y, x, y]
    mapped *= [sx, sy, sx, sy]
    full = np.zeros((oh, ow), np.uint8)
    left, top = round(x * sx), round(y * sy)
    right, bottom = round((x + width) * sx), round((y + height) * sy)
    full[top:bottom, left:right] = cv2.resize(mask.astype(np.uint8), (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
    clipped = origin != "full" and (box[0] < 4 or box[1] < 4 or box[2] > width-4 or box[3] > height-4)
    return Detection(mapped, float(score), full.astype(bool), origin, clipped)


def deduplicate(items, iou=0.65, containment=0.9):
    """Only suppress cross-pass duplicates. Preserve overlapping people within one pass."""
    kept = []
    for item in sorted(items, key=lambda d: (not d.clipped, d.score), reverse=True):
        duplicate = False
        area = int(item.mask.sum())
        if not area:
            continue
        for other in kept:
            if item.origin == other.origin:
                continue
            a, b = item.box, other.box
            if min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1]):
                continue
            intersection = int(np.count_nonzero(item.mask & other.mask))
            other_area = int(other.mask.sum())
            overlap = intersection / (area + other_area - intersection)
            inside = intersection / min(area, other_area)
            # Containment alone would erase a small person inside a merged two-person mask.
            ratio = min(area, other_area) / max(area, other_area)
            if overlap >= iou or (inside >= containment and ratio >= 0.6):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def active_regions(config, camera, clock_s):
    return [r for r in config.get("regions", {}).get(camera, [])
            if r.get("enabled", False) and (not r.get("windows") or any(a <= clock_s < b for a, b in r["windows"]))]


def infer(model, native, camera, clock_s, config, device):
    from mct.perception import PROCESS_SIZE
    nh, nw = native.shape[:2]
    base = cv2.resize(native, PROCESS_SIZE, interpolation=cv2.INTER_AREA)
    results = model.predict(base, imgsz=config["imgsz"], conf=config["conf"], classes=[0],
                            retina_masks=True, device=device, verbose=False)
    items = []
    def add(result, region, source):
        if result.masks is None:
            return
        for box, score, mask in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.masks.data.cpu().numpy()):
            items.append(project_detection(box, score, mask > .5, region, (nw, nh), PROCESS_SIZE, source))
    # Full-frame inference coordinates already use PROCESS_SIZE.
    result = results[0]
    if result.masks is not None:
        for box, score, mask in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.masks.data.cpu().numpy()):
            items.append(Detection(box, float(score), mask > .5, "full"))
    for number, region in enumerate(active_regions(config, camera, clock_s)):
        x1, y1, x2, y2 = region["xyxy"]
        if not (0 <= x1 < x2 <= nw and 0 <= y1 < y2 <= nh):
            raise ValueError(f"Invalid ROI for {camera}: {region}")
        prediction = model.predict(native[y1:y2, x1:x2], imgsz=config["imgsz"], conf=config["conf"],
                                   classes=[0], retina_masks=True, device=device, verbose=False)[0]
        add(prediction, (x1, y1, x2-x1, y2-y1), f"roi:{number}")
    return base, deduplicate(items, **config.get("dedup", {}))
