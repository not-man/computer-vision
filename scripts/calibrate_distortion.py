"""Дисторсия объектива без калибровочного шаблона: подбор модели, при которой края на фоне становятся прямыми."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from mct.geometry import IMAGE_SIZE, Lens, edge_chains, straightness

# Камеры 1, 2, 3 и 5 одной модели (одинаковые часы на кадре и формат записи), поэтому объектив у них общий
LENS_GROUPS = [["camera_new_1", "camera_new_2", "camera_new_3", "camera_new_5"], ["camera_new_4"]]
STRAIGHTNESS_CAP = 0.012  # отношение отклонения к длине; всё, что хуже, считается непрямым предметом


def cost(lens: Lens, chains: list[np.ndarray], weights: np.ndarray) -> float:
    if not lens.is_valid():
        return 1e3
    ratios = np.array([straightness(lens.undistort(chain))[0] for chain in chains])
    return float((weights * np.minimum(ratios, STRAIGHTNESS_CAP)).sum() / weights.sum())


def fit(chains: list[np.ndarray]) -> tuple[Lens, dict]:
    weights = np.array([straightness(chain)[1] for chain in chains])
    identity = cost(Lens(0.0), chains, weights)
    grid = np.arange(-0.95, 0.11, 0.01)
    scores = [cost(Lens(float(k1)), chains, weights) for k1 in grid]
    best_k1 = float(grid[int(np.argmin(scores))])

    # Смещение центра ограничено: при слабых данных он не должен уезжать
    width, height = IMAGE_SIZE
    limit = 0.06 * width

    def objective(x: np.ndarray) -> float:
        if abs(x[2]) > limit or abs(x[3]) > limit:
            return 1e3
        return cost(Lens(float(x[0]), float(x[1]), width / 2 + float(x[2]), height / 2 + float(x[3])), chains, weights)

    start = np.array([best_k1, 0.0, 0.0, 0.0])
    simplex = np.vstack([start, start + np.diag([0.05, 0.05, 15.0, 15.0])])
    result = minimize(
        objective, start, method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 600, "initial_simplex": simplex},
    )
    x = result.x
    lens = Lens(float(x[0]), float(x[1]), width / 2 + float(x[2]), height / 2 + float(x[3]))
    final_ratios = np.array([straightness(lens.undistort(chain))[0] for chain in chains])
    straight = final_ratios < STRAIGHTNESS_CAP / 3
    return lens, {
        "chains": len(chains),
        "k1_grid": round(best_k1, 2),
        "cost_identity": round(identity, 5),
        "cost_k1_only": round(min(scores), 5),
        "cost_final": round(float(result.fun), 5),
        "straight_chains": int(straight.sum()),
        "median_straightness": round(float(np.median(final_ratios[straight])), 5) if straight.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backgrounds", type=Path, default=Path("artifacts/geometry/background"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/geometry"))
    args = parser.parse_args()
    rectified_dir = args.out / "rectified"
    rectified_dir.mkdir(parents=True, exist_ok=True)

    images = {path.stem: cv2.imread(str(path)) for path in sorted(args.backgrounds.glob("camera_new_*.png"))}
    chains = {name: edge_chains(image) for name, image in images.items()}
    lenses = {}
    for group in LENS_GROUPS:
        lens, report = fit([chain for name in group for chain in chains[name]])
        print(group, f"k1={lens.k1:.4f} k2={lens.k2:.4f} cx={lens.cx:.1f} cy={lens.cy:.1f}", json.dumps(report), flush=True)
        for name in group:
            weights = np.array([straightness(chain)[1] for chain in chains[name]])
            print(f"  {name}: цепочек {len(chains[name])}, было {cost(Lens(0.0), chains[name], weights):.5f}, стало {cost(lens, chains[name], weights):.5f}", flush=True)
            lenses[name] = {"k1": round(lens.k1, 5), "k2": round(lens.k2, 5), "cx": round(lens.cx, 2), "cy": round(lens.cy, 2)}
            cv2.imwrite(str(rectified_dir / f"{name}.jpg"), lens.rectify(images[name], 0.6, (1296, 972)))
    (args.out / "lenses.json").write_text(json.dumps(lenses, indent=2))


if __name__ == "__main__":
    main()
