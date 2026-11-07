"""Собирает архив для загрузки на Google Drive: код и настройки, с --full ещё записи и готовая детекция."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["pyproject.toml", "README.md", "configs", "scripts", "src", "evaluation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="добавить записи и готовую детекцию: mct_colab.zip, в git не кладётся")
    args = parser.parse_args()

    out = ROOT / "artifacts" / ("mct_colab.zip" if args.full else "mct_code.zip")
    out.parent.mkdir(exist_ok=True)
    files = [file for name in CODE for file in ([ROOT / name] if (ROOT / name).is_file() else sorted((ROOT / name).rglob("*")))]
    if args.full:
        files += sorted((ROOT / "data").glob("*.mkv"))
        perception = ROOT / "artifacts" / "perception"
        files += sorted(perception.glob("camera_new_?.parquet")) + sorted(perception.glob("*_emb.npz"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if file.is_file() and "__pycache__" not in file.parts:
                # записи уже сжаты: повторное сжатие только тратит время
                archive.write(file, file.relative_to(ROOT), zipfile.ZIP_STORED if file.suffix == ".mkv" else zipfile.ZIP_DEFLATED)
    print(out, round(out.stat().st_size / 1e6), "МБ")


if __name__ == "__main__":
    main()
