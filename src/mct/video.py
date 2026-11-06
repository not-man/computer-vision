"""Чтение и запись видео через ffmpeg из imageio-ffmpeg: системный ffmpeg не требуется."""
import math
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def clock_text(clock_s: float) -> str:
    """Секунды от 17:00:00 в показания часов на кадре."""
    tenths = math.floor((17 * 3600 + clock_s) * 10 + 1e-6)  # вниз, как часы: 59.96 это ещё 59.9, а не 60.0
    return f"{tenths // 36000:02d}:{tenths % 36000 // 600:02d}:{tenths % 600 / 10:04.1f}"


def sampled_frame_index(k: int, step: float) -> int:
    """Номер исходного кадра, который read_frames отдаёт k-м (без учёта начала отрезка)."""
    return math.ceil(k * step - 1e-9)


def read_frames(
    path: Path,
    size: tuple[int, int],
    source_fps: float,
    target_fps: float,
    start_s: float = 0.0,
    duration_s: float | None = None,
    threads: int = 3,
) -> Iterator[tuple[int, np.ndarray]]:
    """Отдаёт (номер исходного кадра, кадр BGR). Выборка по номеру кадра, а не по меткам времени файла.

    start_s и duration_s отсчитываются как frame / source_fps. Декодирование идёт от начала файла:
    перемотка опирается на метки файла и попадает не туда. Пропущенные кадры не масштабируются.
    """
    width, height = size
    step = source_fps / target_fps
    first_k = math.ceil(start_s * target_fps)
    last_k = math.inf if duration_s is None else first_k + round(duration_s * target_fps) - 1
    first_frame = sampled_frame_index(first_k, step)
    filters = [f"select='gte(n\\,{first_frame})*lt(mod(n\\,{step})\\,1)'", f"scale={width}:{height}"]
    cmd = [
        FFMPEG, "-v", "fatal", "-threads", str(threads), "-i", str(path), "-map", "0:v",
        "-fps_mode", "passthrough", "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        k = first_k
        while k <= last_k and len(buf := proc.stdout.read(frame_bytes)) == frame_bytes:
            yield sampled_frame_index(k, step), np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_selected_frames(path: Path, frames: Iterable[int], size: tuple[int, int]) -> Iterator[tuple[int, np.ndarray]]:
    """Отдаёт только указанные кадры по возрастанию номеров. Остальные декодируются, но не преобразуются.

    Кадры считаются подряд, как в read_frames: на метки времени и перемотку файла здесь полагаться нельзя.
    """
    capture = cv2.VideoCapture(str(path))
    position = 0
    try:
        for number in sorted({int(value) for value in frames}):
            while position < number and capture.grab():
                position += 1
            ok, frame = capture.read()
            if not ok or position != number:
                return
            position += 1
            yield number, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    finally:
        capture.release()


class H264Writer:
    """Пишет кадры BGR в MP4, который играет браузер и быстро перематывается."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, bitrate: str = "2M") -> None:
        width, height = size
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG, "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
            "-g", str(round(fps)), "-movflags", "+faststart", "-an", str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg завершился с ошибкой при записи видео")

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
