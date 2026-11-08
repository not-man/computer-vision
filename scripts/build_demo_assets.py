"""Build demo media from existing tracks. No detection, ReID, linking, or threshold sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("render/full"))
    args = parser.parse_args()
    os.chdir(ROOT)
    videos, out = args.videos.resolve(), args.out.resolve()
    if out == videos or out in videos.parents or videos in out.parents:
        parser.error("Output must be separate from input videos")
    out.mkdir(parents=True, exist_ok=True)
    inputs = sorted((ROOT / "configs").glob("*.yaml")) + sorted((ROOT / "src/mct").glob("*.py"))
    inputs += sorted((ROOT / "scripts").glob("*.py"))
    inputs += [ROOT / "artifacts/tracks" / name for name in ("tracks.json", "observations.parquet", "robustness.csv", "tracks.csv")]
    inputs += sorted((ROOT / "artifacts/tracks/crops").glob("*"))
    inputs += [ROOT / "data/office.jpg", ROOT / "results/tracks.csv"]
    for camera in range(1, 6):
        inputs += [videos / f"camera_new_{camera}.mkv", ROOT / f"artifacts/perception/camera_new_{camera}.parquet"]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing))
    fingerprint = hashlib.sha256(json.dumps([(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name, digest(p))
                                             for p in inputs], sort_keys=True).encode()).hexdigest()
    state_file = out / "demo_build_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    if state and state.get("fingerprint") != fingerprint:
        parser.error("Inputs changed; choose a new --out directory to preserve previous results")
    state = state or {"fingerprint": fingerprint, "stages": {}}
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    stages = [
        ("mosaic", ["render_mosaic.py", "artifacts/perception", "artifacts/tracks", "--videos", str(videos), "--out", str(out)]),
        ("showcase", ["make_showcase.py", "--render", str(out)]),
        ("report", ["make_report.py", "--render", str(out)]),
        ("check", ["demo_check.py", "--render", str(out)]),
    ]
    for stage_index, (name, command) in enumerate(stages):
        previous = state["stages"].get(name)
        if name != "check" and previous and all((out / p).is_file() and digest(out / p) == h for p, h in previous["files"].items()):
            print(f"{name}: verified, skipping", flush=True)
            continue
        print(f"{name}: starting", flush=True)
        # Rebuilding an upstream stage invalidates every later stage, including after interruption.
        for later, _ in stages[stage_index:]:
            state["stages"].pop(later, None)
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        started = time.perf_counter()
        with (out / f"build_{name}.log").open("w", encoding="utf-8") as log:
            run = subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]], stdout=log, stderr=subprocess.STDOUT, env=env)
        if run.returncode:
            print((out / f"build_{name}.log").read_text(encoding="utf-8"), flush=True)
            raise SystemExit(run.returncode)
        outputs = {"mosaic": [out / "manifest.json", *sorted((out / "chunks").glob("*.mp4"))],
                   "showcase": [out / "showcase.mp4"],
                   "report": [out / "report.html", out / "plan.jpg", out / "built_from.json"],
                   "check": [out / "build_check.log"]}[name]
        state["stages"][name] = {"seconds": round(time.perf_counter()-started, 2), "files": {str(p.relative_to(out)): digest(p) for p in outputs}}
        temp = state_file.with_suffix(".pending")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temp, state_file)
        print(f"{name}: passed in {state['stages'][name]['seconds']}s", flush=True)
    print("COMPLETE: " + str(out), flush=True)


if __name__ == "__main__":
    main()
