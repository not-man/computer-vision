"""Собирает запасной офлайн-отчёт рядом с роликами: render/full/report.html."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from mct.demo_check import source_stamp
from mct.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--render", type=Path, default=Path("render/full"))
    parser.add_argument("--plan", type=Path, default=Path("data/office.jpg"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    args = parser.parse_args()

    plan_image = args.render / "plan.jpg"
    if not plan_image.exists():
        shutil.copy(args.plan, plan_image)
    build_report(
        json.loads((args.tracks / "tracks.json").read_text()),
        json.loads((args.render / "manifest.json").read_text()),
        pd.read_csv(args.tracks / "robustness.csv"),
        args.tracks, plan_image, args.configs, args.render / "report.html",
    )
    (args.render / "built_from.json").write_text(json.dumps(source_stamp(args.tracks), indent=1))
    print(f"Отчёт: {(args.render / 'report.html').stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
