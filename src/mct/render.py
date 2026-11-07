"""Рендер мозаики: все камеры по общим часам, силуэты цветом трека, живой план и плашки событий.

Читает только готовые файлы. Части по минуте играют в браузере без загрузки всей записи.
"""
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

from mct.geometry import IMAGE_SIZE, load_cameras
from mct.plan import Plan
from mct.video import H264Writer, clock_text, read_frames

CANVAS = (1280, 720)
CELL = (426, 320)
FPS = 10.0
CELL_ORIGINS = [(0, 0), (427, 0), (854, 0), (0, 320), (427, 320)]
PLAN_ORIGIN = (854, 320)
PLAN_SIZE = (172, 400)
PANEL_ORIGIN = (1030, 320)
PANEL_SIZE = (250, 400)
STRIP_ORIGIN = (0, 640)
STRIP_SIZE = (853, 80)
PLATES_ORIGIN = (150, 640)
PLATES_SIZE = (703, 80)

PLAN_CROP = (296, 240, 680, 1130)  # пиксели размеченного плана: зал, коридор, боковой проход
TRAIL_S = 12.0
PATH_GAP_S = 1.0  # дальше этого точки пути не соединяются сплошной линией
DASH_PX = {"occluded": 3.0, "blind_inferred": 7.0}  # два смысла пунктира из ТЗ и со слайда 7
CLOSED_VISIBLE_S = 3.0
PLATE_S = 4.0
PANEL_ROWS = 9
ADJACENT_GAP_S = 0.2  # разрыв короче этого на плашке считается передачей встык
# Цвета с заметной разницей и на тёмном, и на светлом фоне, порядок BGR
PALETTE = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245), (180, 30, 145),
    (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250), (128, 128, 0), (40, 110, 170),
]
COLOR_PAUSE_S = CLOSED_VISIBLE_S + 5.0  # цвет ушедшего трека не достаётся новому, пока след ещё на плане
MASK_ALPHA = 0.45
BACKGROUND = (28, 28, 28)
TEXT = (235, 235, 235)
MUTED = (165, 165, 165)
GREEN = (90, 200, 90)
YELLOW = (40, 200, 240)
BLUE = (235, 170, 80)
GHOST = (220, 150, 220)
CLOSE_TEXT = {
    "DOOR": "ушёл в дверь",
    "STATIONARY": "остался на месте и пропал",
    "OCCLUDER": "скрылся за препятствием",
    "EDGE_LINK": "ушёл в слепую зону",
    "EDGE_OUT": "вышел из кадра",
    "OPEN": "потерян на открытом месте",
}


def assign_colors(tracks: list[dict]) -> dict[int, tuple[int, int, int]]:
    """Цвет на весь трек: берём цвет, освободившийся раньше всех, поэтому одновременные треки его не делят."""
    released = [-np.inf] * len(PALETTE)
    colors = {}
    for track in sorted(tracks, key=lambda x: (x["t_start_ms"], x["track_id"])):
        choice = int(np.argmin(released))
        released[choice] = track["t_end_ms"] / 1000 + COLOR_PAUSE_S
        colors[track["track_id"]] = PALETTE[choice]
    return colors


def short_camera(name: str) -> str:
    return f"cam_{name[-1]}"


class CameraStream:
    """Кадры одной камеры по общему времени. Запросы идут по возрастанию, поэтому чтение последовательное."""

    def __init__(self, frames: Iterator[tuple[int, np.ndarray]], clock_at_frame_0: float, fps: float, tolerance_s: float = 0.1) -> None:
        self._frames = frames
        self._clock0 = clock_at_frame_0
        self._fps = fps
        self._tolerance = tolerance_s
        self._current = next(frames, None)
        self._upcoming = next(frames, None)

    def _clock_of(self, item: tuple[int, np.ndarray]) -> float:
        return self._clock0 + item[0] / self._fps

    def at(self, clock_s: float) -> tuple[int, np.ndarray] | None:
        """Ближайший к моменту кадр или None, если в это время записи нет."""
        while self._upcoming is not None and abs(self._clock_of(self._upcoming) - clock_s) <= abs(self._clock_of(self._current) - clock_s):
            self._current, self._upcoming = self._upcoming, next(self._frames, None)
        if self._current is None or abs(self._clock_of(self._current) - clock_s) > self._tolerance:
            return None
        return self._current


@dataclass(frozen=True)
class Silhouette:
    track: int | None  # None: обломок без итогового трека
    rle: str
    box: tuple[float, float, float, float]


def load_silhouettes(perception: Path, camera: str, observations: pd.DataFrame, track_of: dict[int, int]) -> dict[int, list[Silhouette]]:
    """Силуэты по номерам кадров. Отброшенные фильтром зеркала и экраны сюда не попадают."""
    path = perception / f"{camera}.parquet"
    if not path.exists():
        return {}
    detections = pd.read_parquet(path, columns=["frame", "raw_track", "x1", "y1", "x2", "y2", "mask_rle"])
    assigned = observations.loc[observations.camera == camera, ["frame", "raw_track", "tracklet"]]
    merged = detections.merge(assigned, on=["frame", "raw_track"], how="inner")
    silhouettes: dict[int, list[Silhouette]] = {}
    for row in merged.itertuples():
        track = track_of.get(int(row.tracklet))
        silhouettes.setdefault(int(row.frame), []).append(
            Silhouette(track, row.mask_rle, (row.x1, row.y1, row.x2, row.y2))
        )
    return silhouettes


def _scaled_mask(silhouette: Silhouette) -> tuple[np.ndarray, int, int] | None:
    """Маска в размере ячейки только в пределах рамки: так быстрее, чем масштабировать весь кадр."""
    width, height = IMAGE_SIZE
    x1, y1, x2, y2 = silhouette.box
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
    if x2 <= x1 or y2 <= y1:
        return None
    full = mask_utils.decode({"size": [height, width], "counts": silhouette.rle.encode()})
    sx, sy = CELL[0] / width, CELL[1] / height
    left, top = int(x1 * sx), int(y1 * sy)
    size = (max(int(round((x2 - x1) * sx)), 1), max(int(round((y2 - y1) * sy)), 1))
    return cv2.resize(full[y1:y2, x1:x2], size, interpolation=cv2.INTER_NEAREST), left, top


def paint_silhouette(cell: np.ndarray, silhouette: Silhouette, color: tuple[int, int, int] | None) -> tuple[int, int, int, int] | None:
    """Рисует силуэт на ячейке; цвет None: только серый контур. Возвращает прямоугольник на ячейке."""
    scaled = _scaled_mask(silhouette)
    if scaled is None:
        return None
    mask, left, top = scaled
    height, width = mask.shape
    width, height = min(width, cell.shape[1] - left), min(height, cell.shape[0] - top)
    if width <= 0 or height <= 0:
        return None
    mask = mask[:height, :width]
    region = cell[top:top + height, left:left + width]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if color is None:
        cv2.drawContours(region, contours, -1, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        inside = mask > 0
        region[inside] = ((1 - MASK_ALPHA) * region[inside] + MASK_ALPHA * np.array(color)).astype(np.uint8)
        cv2.drawContours(region, contours, -1, color, 1, cv2.LINE_AA)
    return left, top, width, height


def draw_tag(image: np.ndarray, text: str, left: int, top: int, color: tuple[int, int, int], taken: list[tuple[int, int, int, int]], scale: float = 0.5) -> None:
    """Номер трека на цветной подложке внутри изображения; соседние подписи не закрывают друг друга."""
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    box_width, box_height = width + 6, height + 6
    left = int(np.clip(left, 0, image.shape[1] - box_width))
    top = int(np.clip(top - box_height, 0, image.shape[0] - box_height))
    for x, y, w, h in sorted(taken):
        if left < x + w and x < left + box_width and top < y + h and y < top + box_height:
            left = x + w + 2
    if left + box_width > image.shape[1]:
        left = max(min(x for x, _, _, _ in taken) - box_width - 2, 0)
    taken.append((left, top, box_width, box_height))
    cv2.rectangle(image, (left, top), (left + box_width, top + box_height), color, -1)
    ink = (0, 0, 0) if sum(color) > 350 else (255, 255, 255)
    cv2.putText(image, text, (left + 3, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)


class Stamps:
    """Кириллица через Pillow; готовые надписи кэшируются, поэтому она не тормозит рендер."""

    def __init__(self) -> None:
        fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        self._regular = str(fonts / "DejaVuSans.ttf")
        self._bold = str(fonts / "DejaVuSans-Bold.ttf")
        self._fonts: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}
        self._cache: dict[tuple, np.ndarray] = {}

    def _font(self, bold: bool, size: int) -> ImageFont.FreeTypeFont:
        key = (bold, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(self._bold if bold else self._regular, size)
        return self._fonts[key]

    def text(self, text: str, size: int = 15, color: tuple[int, int, int] = TEXT, bold: bool = False, max_width: int | None = None,
             background: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
        key = (text, size, color, bold, max_width, background)
        if key in self._cache:
            return self._cache[key]
        font = self._font(bold, size)
        lines = _wrap(text, font, max_width) if max_width else [text]
        line_height = size + 4
        width = int(max(font.getlength(line) for line in lines)) + 2
        canvas = Image.new("RGB", (max(width, 1), line_height * len(lines)), background[::-1])
        draw = ImageDraw.Draw(canvas)
        for index, line in enumerate(lines):
            draw.text((0, index * line_height), line, font=font, fill=color[::-1])
        stamp = np.asarray(canvas)[:, :, ::-1].copy()
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = stamp
        return stamp


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current or not lines else lines


def paste(image: np.ndarray, stamp: np.ndarray, left: int, top: int) -> None:
    height = min(stamp.shape[0], image.shape[0] - top)
    width = min(stamp.shape[1], image.shape[1] - left)
    if height > 0 and width > 0:
        image[top:top + height, left:left + width] = stamp[:height, :width]


@dataclass(frozen=True)
class Event:
    t: float
    kind: str  # handoff, fast, resume, close, ghost
    text: str
    color: tuple[int, int, int]
    track_id: int
    old_track_id: int | None = None


def build_events(result: dict, recording_end_s: float | None = None) -> list[Event]:
    """Плашки видео: передачи между камерами, разрывы, закрытия и подсказки Призрака. Только из готовых треков."""
    events = []
    for track in result["tracks"]:
        track_id = track["track_id"]
        for link in track["links"]:
            t = link["t_ms"] / 1000
            color = YELLOW if link["confidence"] == "yellow" else GREEN
            r = "" if link["r"] is None else f"  R {link['r']:.2f}"
            route = f"{short_camera(link['camera_a'])} → {short_camera(link['camera_b'])}"
            if link["tier"] == "sustained":
                mark = "  спорная" if link["confidence"] == "yellow" else ""
                events.append(Event(t, "handoff", f"#{track_id}  {route}  G {link['g_m']:.1f} м  T {link['contact_s']:.1f} с{r}{mark}", color, track_id))
            elif link["tier"] == "brief":
                contact = f"контакт {link['contact_s']:.1f} с" if link["contact_s"] >= 0 else f"разрыв {-link['contact_s']:.1f} с"
                events.append(Event(t, "fast", f"#{track_id}  {route}  быстрый переход  {contact}  G {link['g_m']:.1f} м{r}", BLUE, track_id))
            elif link["tier"] == "return":
                absent = int(round(-link["contact_s"]))
                events.append(Event(t - link["contact_s"], "resume", f"#{track_id}  вернулся через {absent // 60}:{absent % 60:02d}, номер сохранён{r}", GHOST, track_id))
            elif -link["contact_s"] < ADJACENT_GAP_S:
                events.append(Event(t, "handoff", f"#{track_id}  {route}  передача встык  G {link['g_m']:.1f} м{r}", color, track_id))
            else:
                where = f"на {short_camera(link['camera_b'])}" if link["camera_a"] == link["camera_b"] else route
                events.append(Event(t, "resume", f"#{track_id}  скрыт {-link['contact_s']:.1f} с и продолжен {where}  G {link['g_m']:.1f} м{r}", BLUE, track_id))

        end = track["t_end_ms"] / 1000
        if recording_end_s is None or end < recording_end_s - 1.0:
            reason = CLOSE_TEXT.get(track["close_zone"], track["close_zone"])
            events.append(Event(end, "close", f"Трек #{track_id} закрыт: {reason}", MUTED, track_id))

        for hint in track["possibly_same_person_as"]:
            absent = int(round(hint["absent_s"]))
            text = (
                f"НОВЫЙ ТРЕК #{track_id} · Вероятно, вернулся #{hint['track_id']} · отсутствовал {absent // 60}:{absent % 60:02d} · "
                f"сходство {hint['r']:.2f} · не склеено по правилу ТЗ"
            )
            events.append(Event(track["t_start_ms"] / 1000, "ghost", text, GHOST, track_id, hint["track_id"]))
    events.sort(key=lambda e: (e.t, e.track_id, e.kind))
    return events


@dataclass
class TrackView:
    """Положение трека на плане во времени."""

    track_id: int
    color: tuple[int, int, int]
    times: np.ndarray
    xy: np.ndarray
    t_start: float
    t_end: float
    cameras_at: list[tuple[str, float, float]] = field(default_factory=list)
    pauses: list[tuple[str, float, float]] = field(default_factory=list)

    @classmethod
    def from_json(cls, track: dict, color: tuple[int, int, int] = (255, 255, 255)) -> "TrackView":
        path = np.array(track["plan_path"], np.float64).reshape(-1, 3)
        spans = [(s["kind"], s["camera_id"], s["t_start_ms"] / 1000, s["t_end_ms"] / 1000) for s in track["segments"]]
        return cls(track["track_id"], color, path[:, 2] / 1000, path[:, :2], track["t_start_ms"] / 1000, track["t_end_ms"] / 1000,
                   [(camera, start, end) for kind, camera, start, end in spans if kind == "observed"],
                   [(kind, start, end) for kind, _, start, end in spans if kind != "observed"])

    def position(self, t: float) -> tuple[np.ndarray, bool] | None:
        """Положение и признак «сейчас скрыт»: во время паузы показываем последнюю известную точку."""
        if not len(self.times) or t < self.times[0] or t > self.times[-1]:
            return None
        right = int(np.searchsorted(self.times, t))
        if right == 0 or self.times[right] == t:
            return self.xy[right], False
        left = right - 1
        gap = self.times[right] - self.times[left]
        if gap > PATH_GAP_S:
            return self.xy[left], True
        return self.xy[left] + (self.xy[right] - self.xy[left]) * (t - self.times[left]) / gap, False

    def _pause_kind(self, t: float) -> str:
        return next((kind for kind, start, end in self.pauses if start <= t <= end), "occluded")

    def trail(self, t: float, length_s: float = TRAIL_S) -> list[tuple[np.ndarray, str | None]]:
        """Участки пути за последние секунды: (точки, вид паузы). None: сплошная, человека видели."""
        mask = (self.times >= t - length_s) & (self.times <= t)
        points, times = self.xy[mask], self.times[mask]
        if len(points) == 0:
            return []
        current = self.position(t)
        if current is not None and not current[1]:
            points, times = np.vstack([points, current[0]]), np.append(times, t)
        pieces, start = [], 0
        for index in range(1, len(times)):
            if times[index] - times[index - 1] > PATH_GAP_S:
                pieces.append((points[start:index], None))
                pieces.append((points[index - 1:index + 1], self._pause_kind((times[index - 1] + times[index]) / 2)))
                start = index
        pieces.append((points[start:], None))
        return [piece for piece in pieces if len(piece[0]) >= 2]

    def cameras_now(self, t: float) -> list[str]:
        return sorted({camera[-1] for camera, start, end in self.cameras_at if start <= t <= end})


def draw_dashed(image: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], dash: float = 5.0, thickness: int = 1) -> None:
    span = b - a
    length = float(np.linalg.norm(span))
    if length < 1:
        return
    direction = span / length
    position = 0.0
    while position < length:
        start = a + direction * position
        end = a + direction * min(position + dash, length)
        cv2.line(image, tuple(start.round().astype(int).tolist()), tuple(end.round().astype(int).tolist()), color, thickness, cv2.LINE_AA)
        position += 2 * dash


class LivePlan:
    """Живой план: неподвижная подложка и текущие положения всех треков."""

    def __init__(self, plan_image: Path, plan: Plan, image_scale: float, image_offset: tuple[float, float]) -> None:
        self._px_per_m = plan.px_per_m
        x0, y0, x1, y1 = PLAN_CROP
        self._origin = np.array([x0, y0], np.float64)
        self._scale = np.array([PLAN_SIZE[0] / (x1 - x0), PLAN_SIZE[1] / (y1 - y0)])
        # Чистый план без значков камер: его координаты пересчитываются в координаты разметки
        source = cv2.imread(str(plan_image))
        matrix = np.array([[self._scale[0] * image_scale, 0, self._scale[0] * (image_offset[0] - x0)],
                           [0, self._scale[1] * image_scale, self._scale[1] * (image_offset[1] - y0)]])
        warped = cv2.warpAffine(source, matrix, PLAN_SIZE, flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
        self.base = cv2.addWeighted(warped, 0.55, np.full_like(warped, 255), 0.45, 0)

    def to_pixels(self, xy_m: np.ndarray) -> np.ndarray:
        return (np.asarray(xy_m, np.float64) * self._px_per_m - self._origin) * self._scale

    def add_camera(self, label: str, xy_m: np.ndarray, yaw: float) -> None:
        center = self.to_pixels(xy_m)
        tip = center + 11 * np.array([np.cos(yaw), np.sin(yaw) * self._scale[1] / self._scale[0]])
        cv2.arrowedLine(self.base, tuple(center.round().astype(int).tolist()), tuple(tip.round().astype(int).tolist()), (60, 60, 60), 1, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(self.base, tuple(center.round().astype(int).tolist()), 6, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.putText(self.base, label, tuple((center + (-3, 3)).round().astype(int).tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def draw(self, views: list[TrackView], t: float) -> np.ndarray:
        image = self.base.copy()
        for view in views:
            if t < view.t_start or t > view.t_end + CLOSED_VISIBLE_S:
                continue
            for points, pause in view.trail(t):
                pixels = self.to_pixels(points)
                if pause is not None:
                    draw_dashed(image, pixels[0], pixels[-1], view.color, DASH_PX.get(pause, DASH_PX["occluded"]))
                else:
                    cv2.polylines(image, [pixels.round().astype(np.int32).reshape(-1, 1, 2)], False, view.color, 2, cv2.LINE_AA)
            position = view.position(min(t, view.t_end))
            if position is None or t > view.t_end:
                continue
            center = tuple(self.to_pixels(position[0]).round().astype(int).tolist())
            if position[1]:
                cv2.circle(image, center, 4, view.color, 1, cv2.LINE_AA)
            else:
                cv2.circle(image, center, 4, view.color, -1, cv2.LINE_AA)
                cv2.circle(image, center, 4, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(image, str(view.track_id), (center[0] + 6, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        return image


def chunk_index(t: float, start: float, chunk_s: float) -> int:
    return int((t - start + 1e-6) // chunk_s)


def ticks(start: float, end: float, fps: float = FPS) -> np.ndarray:
    """Моменты кадров мозаики: равномерная сетка по общему времени."""
    return start + np.arange(int(round((end - start) * fps))) / fps


def _draw_panel(image: np.ndarray, stamps: Stamps, started: int, total: int, hints: int, active: list[TrackView], t: float, clock_note: str) -> None:
    left, top = PANEL_ORIGIN
    image[top:top + PANEL_SIZE[1], left:left + PANEL_SIZE[0]] = BACKGROUND
    paste(image, stamps.text("Треков по ТЗ", 13, MUTED), left + 8, top + 6)
    count = stamps.text(str(started), 30, TEXT, bold=True)
    paste(image, count, left + 8, top + 24)
    paste(image, stamps.text(f"из {total} за запись", 12, MUTED), left + 14 + count.shape[1], top + 42)
    paste(image, stamps.text(f"подсказок о возврате: {hints}", 11, GHOST), left + 8, top + 64)
    paste(image, stamps.text(f"Сейчас на камерах: {len(active)}", 13, TEXT), left + 8, top + 84)
    row = top + 106
    for view in active[:PANEL_ROWS]:
        cv2.rectangle(image, (left + 8, row + 2), (left + 20, row + 14), view.color, -1)
        cameras = ",".join(view.cameras_now(t)) or "скрыт"
        elapsed = int(t - view.t_start)
        paste(image, stamps.text(f"#{view.track_id}  cam {cameras}  {elapsed // 60}:{elapsed % 60:02d}", 12, TEXT), left + 26, row)
        row += 19
    if len(active) > PANEL_ROWS:
        paste(image, stamps.text(f"и ещё {len(active) - PANEL_ROWS}", 12, MUTED), left + 26, row)
    for index, line in enumerate(("сплошная: видят камеры", "мелкий пунктир: за препятствием", "крупный пунктир: вне обзора", "G план · T контакт · R внешность")):
        paste(image, stamps.text(line, 11, MUTED), left + 8, top + PANEL_SIZE[1] - 92 + index * 15)
    paste(image, stamps.text(clock_note, 11, MUTED, max_width=PANEL_SIZE[0] - 14), left + 8, top + PANEL_SIZE[1] - 31)


def _draw_camera_label(cell: np.ndarray, stamps: Stamps, name: str) -> None:
    stamp = stamps.text(short_camera(name), 12, (0, 255, 255), background=(0, 0, 0))
    paste(cell, stamp, 3, CELL[1] - stamp.shape[0] - 3)


def _fit_height(crop: np.ndarray, height: int) -> np.ndarray:
    scale = height / max(crop.shape[0], 1)
    return cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), height), interpolation=cv2.INTER_LINEAR)


def _draw_strip(image: np.ndarray, stamps: Stamps, events: list[Event], crops: dict[int, np.ndarray], t: float) -> None:
    left, top = STRIP_ORIGIN
    image[top:top + STRIP_SIZE[1], left:left + STRIP_SIZE[0]] = BACKGROUND
    paste(image, stamps.text(clock_text(t)[:8], 24, TEXT, bold=True), left + 8, top + 12)
    paste(image, stamps.text("общее время", 11, MUTED), left + 8, top + 46)

    plate_left, plate_top = PLATES_ORIGIN
    recent = [e for e in events if e.t <= t < e.t + PLATE_S]
    ghost = next((e for e in reversed(recent) if e.kind == "ghost"), None)
    if ghost is not None:
        x = plate_left
        for track_id in (ghost.old_track_id, ghost.track_id):
            if track_id in crops:
                thumb = _fit_height(crops[track_id], 72)
                paste(image, thumb, x, plate_top + 4)
                x += thumb.shape[1] + 4
        cv2.rectangle(image, (x, plate_top + 4), (x + 3, plate_top + 76), ghost.color, -1)
        paste(image, stamps.text(ghost.text, 14, TEXT, bold=True, max_width=PLATES_SIZE[0] - (x - plate_left) - 12), x + 8, plate_top + 10)
        return
    row = plate_top + 4
    for event in recent[-3:]:
        cv2.rectangle(image, (plate_left, row), (plate_left + 3, row + 20), event.color, -1)
        paste(image, stamps.text(event.text, 14, TEXT, max_width=PLATES_SIZE[0] - 14), plate_left + 9, row + 1)
        row += 25


def render_mosaic(
    videos: Path,
    perception: Path,
    tracks_dir: Path,
    plan_image: Path,
    out_dir: Path,
    clock_start: float | None = None,
    clock_end: float | None = None,
    chunk_s: int = 60,
    show_fragments: bool = True,
    configs: Path = Path("configs"),
) -> dict:
    """Рендерит части мозаики и возвращает манифест для интерфейса."""
    result = json.loads((tracks_dir / "tracks.json").read_text())
    observations = pd.read_parquet(tracks_dir / "observations.parquet", columns=["camera", "frame", "raw_track", "tracklet"])
    track_of = {tracklet: track["track_id"] for track in result["tracks"] for tracklet in track["tracklets"]}
    clocks = yaml.safe_load((configs / "clocks.yaml").read_text())
    features = yaml.safe_load((configs / "plan_features.yaml").read_text())
    cameras = load_cameras()
    names = list(clocks)

    if clock_start is None:
        clock_start = min(c["clock_at_frame_0"] for c in clocks.values())
    if clock_end is None:
        clock_end = max(c["clock_at_frame_0"] + c["frames"] / c["fps"] for c in clocks.values())
    corrections = [c["clock_at_frame_0"] - c["osd_at_frame_0"] for c in clocks.values()]
    # Жюри увидит разные секунды в углах кадров, поэтому объясняем это прямо на видео
    clock_note = f"часы на кадрах расходятся до {max(corrections) - min(corrections):.1f} с; кадры выровнены по движению людей"

    streams, silhouettes = {}, {}
    for name in names:
        clock = clocks[name]
        silhouettes[name] = load_silhouettes(perception, name, observations, track_of)
        skip = max(clock_start - clock["clock_at_frame_0"] - 1.0, 0.0)
        streams[name] = CameraStream(read_frames(videos / f"{name}.mkv", CELL, clock["fps"], FPS, skip, threads=2), clock["clock_at_frame_0"], clock["fps"])

    plan = Plan.load(configs / "plan_features.yaml")
    image_config = features["plan_image"]
    live_plan = LivePlan(plan_image, plan, image_config["scale"], (image_config["dx"], image_config["dy"]))
    for name, camera in cameras.items():
        live_plan.add_camera(name[-1], np.array([camera.x, camera.y]), camera.yaw)

    colors = assign_colors(result["tracks"])
    views = [TrackView.from_json(track, colors[track["track_id"]]) for track in result["tracks"]]
    events = build_events(result, clock_end)
    total = result["total_tracks"]
    ghost_times = np.array([event.t for event in events if event.kind == "ghost"])
    stamps = Stamps()
    crops: dict[int, np.ndarray] = {}
    crop_score: dict[int, float] = {}
    missing = stamps.text("нет записи", 14, MUTED, background=(0, 0, 0))

    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_chunks = []
    writer: H264Writer | None = None
    current_chunk = -1
    chunk_path: Path | None = None
    chunk_started = 0.0

    def finish_chunk() -> None:
        if writer is not None:
            writer.close()
            manifest_chunks[-1]["mb"] = round(chunk_path.stat().st_size / 1e6, 2)
            manifest_chunks[-1]["render_s"] = round(time.perf_counter() - chunk_started, 1)

    for t in ticks(clock_start, clock_end):
        index = chunk_index(t, clock_start, chunk_s)
        if index != current_chunk:
            finish_chunk()
            current_chunk = index
            chunk_started = time.perf_counter()
            chunk_path = chunks_dir / f"{index:03d}.mp4"
            writer = H264Writer(chunk_path, CANVAS, FPS)
            manifest_chunks.append({
                "file": f"chunks/{index:03d}.mp4", "t_start_s": round(float(clock_start + index * chunk_s), 3),
                "t_end_s": round(float(min(clock_start + (index + 1) * chunk_s, clock_end)), 3), "start": clock_text(clock_start + index * chunk_s),
            })

        canvas = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
        for name, (x, y) in zip(names, CELL_ORIGINS):
            got = streams[name].at(float(t))
            cell = canvas[y:y + CELL[1], x:x + CELL[0]]
            if got is None:
                paste(cell, missing, 170, 150)
            else:
                frame_number, frame = got
                cell[:] = frame
                tags = []
                for silhouette in silhouettes[name].get(frame_number, []):
                    if silhouette.track is None and not show_fragments:
                        continue
                    color = None if silhouette.track is None else colors[silhouette.track]
                    rectangle = paint_silhouette(cell, silhouette, color)
                    if rectangle is None or silhouette.track is None:
                        continue
                    tags.append((silhouette.track, rectangle, color))
                    # Снимок для плашки и досье: самый крупный вид человека, не обрезанный краем ячейки
                    left, top, width, height = rectangle
                    area = float(width * height)
                    inner = left > 1 and top > 1 and left + width < CELL[0] - 1 and top + height < CELL[1] - 1
                    if inner and area > crop_score.get(silhouette.track, 0.0):
                        crop_score[silhouette.track] = area
                        crops[silhouette.track] = frame[top:top + height, left:left + width].copy()
                taken: list[tuple[int, int, int, int]] = []
                for track, (left, top, _, _), color in sorted(tags, key=lambda item: item[1][0]):
                    draw_tag(cell, f"#{track}", left, top, color, taken)
            _draw_camera_label(cell, stamps, name)

        active = [view for view in views if view.t_start <= t <= view.t_end]
        started = sum(view.t_start <= t for view in views)
        _draw_strip(canvas, stamps, events, crops, float(t))
        canvas[PLAN_ORIGIN[1]:PLAN_ORIGIN[1] + PLAN_SIZE[1], PLAN_ORIGIN[0]:PLAN_ORIGIN[0] + PLAN_SIZE[0]] = live_plan.draw(views, float(t))
        hints = int(np.searchsorted(ghost_times, t, side="right"))
        _draw_panel(canvas, stamps, started, total, hints, active, float(t), clock_note)
        writer.write(canvas)
    finish_chunk()

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for track_id, crop in crops.items():
        cv2.imwrite(str(crops_dir / f"track_{track_id}.jpg"), crop)
    manifest = {
        "fps": FPS, "size": list(CANVAS), "chunk_s": chunk_s,
        "clock_start_s": float(clock_start), "clock_end_s": float(clock_end),
        "total_tracks": total,
        "events": [{"t_s": round(e.t, 2), "kind": e.kind, "text": e.text, "track_id": e.track_id} for e in events],
        "chunks": manifest_chunks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest
