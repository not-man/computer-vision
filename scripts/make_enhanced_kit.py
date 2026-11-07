"""Self-contained Colab kit: source, baseline perception, pinned YOLO weight, instructions."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for directory in ("src", "scripts", "configs"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [ROOT / "pyproject.toml", ROOT / "docs/COLAB_ENHANCED.md", ROOT / "weights/yolo11m-seg.pt"]
    for camera in range(1, 6):
        files += [ROOT / f"artifacts/perception/camera_new_{camera}{suffix}" for suffix in (".parquet", "_emb.npz", "_summary.json")]
    target = ROOT / "artifacts/mct_enhanced_kit.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
