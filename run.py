"""Portable entry point: python run.py setup|demo|check|all|test."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "demo", "check", "dense", "all", "tracks", "render", "robustness", "test"])
    args = parser.parse_args()
    os.chdir(ROOT)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def call(*command: str) -> None:
        subprocess.run(command, env=env, check=True)

    if args.command == "setup":
        uv = shutil.which("uv")
        local_uv = ROOT / ".tools" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
        if uv is None and local_uv.exists():
            uv = str(local_uv)
        if uv is None:
            parser.error("Install uv first: python -m pip install uv")
        call(uv, "sync", "--extra", "cpu", "--locked")
        return
    if not python.exists():
        parser.error("Run python run.py setup first")

    def script(name: str, *extra: str) -> None:
        call(str(python), "scripts/" + name + ".py", *extra)

    def tracks() -> None:
        script("link_tracks", "artifacts/perception", "--out", "artifacts/tracks")
        (ROOT / "results").mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "artifacts/tracks/tracks.csv", ROOT / "results/tracks.csv")

    def render() -> None:
        script("render_mosaic", "artifacts/perception", "artifacts/tracks", "--out", "render/full")
        script("draw_gallery", "artifacts/perception", "artifacts/tracks")

    if args.command == "dense":
        script("reembed", "data", "--every", "3", "--kinds", "box", "--out", "artifacts/perception", "--suffix", "dense")
    if args.command in ("tracks", "all"):
        tracks()
    if args.command in ("render", "all"):
        render()
    if args.command in ("robustness", "all"):
        script("check_robustness", "artifacts/perception")
    if args.command == "all":
        script("make_showcase")
        script("make_report")
    if args.command in ("check", "demo", "all"):
        script("demo_check")
    if args.command == "demo":
        call(str(python), "-m", "streamlit", "run", "app/demo.py", "--server.headless=true",
             "--server.address=127.0.0.1", "--browser.gatherUsageStats=false")
    if args.command == "test":
        call(str(python), "-m", "pytest", "-q")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
