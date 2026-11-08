"""Small Colab render kit. Original videos are reused from Drive, not bundled."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for directory in ("src", "scripts", "configs", "evaluation", "artifacts/tracks", "results"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += list((ROOT / "artifacts/perception").glob("camera_new_?.parquet"))
    files += [ROOT / "data/office.jpg"]
    if len(list((ROOT / "artifacts/perception").glob("camera_new_?.parquet"))) != 5:
        raise RuntimeError("Five perception files required")
    document = json.loads((ROOT / "artifacts/tracks/tracks.json").read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"base_commit": commit, "total_tracks": document["total_tracks"], "files": {}}
    output = ROOT / "artifacts/mct_demo_build.zip"
    # Source artifacts are already compressed. STORE keeps local CPU usage low.
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        for p in sorted(set(files)):
            name = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            archive.writestr(name, data)
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
        archive.writestr("demo_bundle.json", json.dumps(manifest, indent=2))
    print(output, round(output.stat().st_size / 1e6, 1), "MB", document["total_tracks"], "tracks")


if __name__ == "__main__":
    main()
