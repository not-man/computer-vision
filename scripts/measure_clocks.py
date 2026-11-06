"""Часы на кадре: номер кадра → показания, по всем сменам секунд.

Период и фаза ищутся по концентрации смен, поэтому ложные срабатывания на фоне не мешают. Смены дрожат на
несколько кадров, поэтому целую секунду задаёт показание, прочитанное в середине секунды, а не первый кадр.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from mct.video import FFMPEG

# Цифра единиц секунд в исходном кадре 2592×1944: (x, y, ширина, высота)
DIGIT_BOXES = {"camera_new_4": (2520, 40, 44, 52)}
DEFAULT_BOX = (578, 8, 40, 44)
GLYPH_BRIGHTNESS = 170
MIN_CHANGED_SHARE = 0.06  # доля пикселей цифры, сменивших яркость
MAX_RESIDUAL_FRAMES = 3.0


def find_ticks(video: Path) -> tuple[np.ndarray, int]:
    """Кадры, где меняется рисунок цифры. Сравниваем только яркие пиксели, чтобы фон за цифрой не мешал."""
    x, y, w, h = DIGIT_BOXES.get(video.stem, DEFAULT_BOX)
    cmd = [
        FFMPEG, "-v", "fatal", "-threads", "3", "-i", str(video), "-map", "0:v", "-fps_mode", "passthrough",
        "-vf", f"crop={w}:{h}:{x}:{y}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    previous, index, changed = None, 0, []
    while len(buf := proc.stdout.read(w * h)) == w * h:
        glyph = np.frombuffer(buf, np.uint8) > GLYPH_BRIGHTNESS
        if previous is not None:
            changed.append(float((glyph != previous).mean()))
        previous = glyph
        index += 1
    proc.wait()
    ticks = np.nonzero(np.array(changed) > MIN_CHANGED_SHARE)[0] + 1
    # Смена цифры размазана кодеком на соседние кадры: оставляем первый кадр каждой группы
    return ticks[np.insert(np.diff(ticks) > 3, 0, True)], index


def fit_period(ticks: np.ndarray, nominal: float) -> tuple[float, float]:
    """Период и фаза смен: максимум концентрации фаз на сетке периодов около заявленного."""
    periods = np.arange(nominal * 0.99, nominal * 1.01, 0.0002)
    strengths = [abs(np.exp(2j * np.pi * ticks / period).mean()) for period in periods]
    period = float(periods[int(np.argmax(strengths))])
    phase = (np.angle(np.exp(2j * np.pi * ticks / period).mean()) % (2 * np.pi)) * period / (2 * np.pi)
    return period, float(phase)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    camera = args.video.stem
    clock = yaml.safe_load(Path("configs/clocks.yaml").read_text())[camera]
    ticks, frames = find_ticks(args.video)
    period, phase = fit_period(ticks, clock["fps"])

    seconds = np.round((ticks - phase) / period)
    good = np.abs(ticks - (phase + seconds * period)) <= MAX_RESIDUAL_FRAMES
    slope, intercept = np.polyfit(seconds[good], ticks[good], 1)
    # Дробная часть из смен секунд, целая из показания на кадре: показание берётся как его середина
    read_frame, read_clock = clock["osd_read"]
    fraction = -intercept / slope
    whole = round(read_clock + 0.5 - (fraction + read_frame / slope))
    print(json.dumps({
        "camera": camera,
        "frames": frames,
        "frames_per_clock_s": round(float(slope), 4),
        "osd_at_frame_0": round(float(fraction + whole), 3),
        "ticks_found": len(ticks),
        "ticks_matched": int(good.sum()),
        "ticks_expected": int(round(frames / slope)),
        "residual_frames_p95": round(float(np.percentile(np.abs(ticks[good] - (intercept + slope * seconds[good])), 95)), 2),
    }))


if __name__ == "__main__":
    main()
