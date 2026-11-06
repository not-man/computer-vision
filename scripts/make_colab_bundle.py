"""Собирает mct_code.zip для загрузки на Google Drive: только код и настройки, без видео и весов."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "README.md", "configs", "scripts", "src"]


def main() -> None:
    out = ROOT / "artifacts" / "mct_code.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE:
            path = ROOT / name
            for file in [path] if path.is_file() else sorted(path.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
