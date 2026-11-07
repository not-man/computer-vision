"""Проверка перед показом. Код возврата 1, если что-то не так: с таким набором файлов выходить нельзя."""
import argparse
import sys
from pathlib import Path

from mct.demo_check import run_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--render", type=Path, default=Path("render/full"))
    parser.add_argument("--results", type=Path, default=Path("results/tracks.csv"))
    args = parser.parse_args()

    results = run_checks(args.tracks, args.render, args.results)
    for name, problems in results.items():
        print(f"{'ОШИБКА' if problems else 'ок'}: {name}")
        for problem in problems:
            print(f"    {problem}")
    if any(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
