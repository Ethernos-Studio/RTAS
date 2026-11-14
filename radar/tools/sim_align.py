"""sim 推演 <-> 录像对齐（重写：用相似变换初值 + 迭代仿射/单应，别再用极值四点）。

输出：视频第 0 秒对应推演第几秒（t0）+ 该关单应矩阵 + 残差。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"E:\RTAS\src")   # rtas.sim（推演系统）
sys.stdout.reconfigure(encoding="utf-8")

GAMEDATA = Path(r"E:\RTAS\ArknightsGameData\zh_CN\gamedata")


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    matrix = np.array(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(matrix)
    solution = vt[-1].reshape(3, 3)
    if abs(solution[2, 2]) < 1e-9:
        return np.eye(3)
    return solution / solution[2, 2]


def apply_h(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1))
    projected = np.hstack([points, ones]) @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        result = projected[:, :2] / projected[:, 2:3]
    return np.nan_to_num(result, nan=1e6, posinf=1e6, neginf=-1e6)


def similarity_init(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """按均值/标准差给一个相似变换当初值（比"极值四点"稳）。"""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_std = max(1e-6, float(src.std()))
    dst_std = max(1e-6, float(dst.std()))
    scale = dst_std / src_std
    matrix = np.array([
        [scale, 0.0, dst_mean[0] - scale * src_mean[0]],
        [0.0, scale, dst_mean[1] - scale * src_mean[1]],
        [0.0, 0.0, 1.0],
    ])
    return matrix


def icp_fit(src: np.ndarray, dst: np.ndarray, *, tolerance: float = 60.0,
            iterations: int = 20, model: str = "homography"):
    if len(src) < 4 or len(dst) < 4:
        return None
    matrix = similarity_init(src, dst)
    best_matrix, best_residual, best_inliers = None, float("inf"), 0
    for _ in range(iterations):
        projected = apply_h(matrix, src)
        distances = np.linalg.norm(projected[:, None, :] - dst[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        residuals = distances[np.arange(len(src)), nearest]
        inliers = residuals < tolerance
        if inliers.sum() < 4:
            break
        pairs_src, pairs_dst = src[inliers], dst[nearest[inliers]]
        if model == "homography" and inliers.sum() >= 6:
            matrix = homography(pairs_src, pairs_dst)
        else:
            # 仿射最小二乘（点数少时更稳）
            design = np.hstack([pairs_src, np.ones((len(pairs_src), 1))])
            solution, *_ = np.linalg.lstsq(design, pairs_dst, rcond=None)
            matrix = np.vstack([solution.T, [0.0, 0.0, 1.0]])
        mean_residual = float(residuals[inliers].mean())
        if (int(inliers.sum()), -mean_residual) > (best_inliers, -best_residual):
            best_matrix, best_residual, best_inliers = matrix, mean_residual, int(inliers.sum())
        tolerance = max(25.0, tolerance * 0.85)
    if best_matrix is None:
        return None
    return best_matrix, best_residual, best_inliers


def main() -> int:
    parser = argparse.ArgumentParser(description="sim 推演与录像对齐")
    parser.add_argument("--stage", default="01-07")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--clicks", default="clicks")
    parser.add_argument("--t-max", type=float, default=150.0)
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()

    from rtas.sim.engine import simulate
    from rtas.sim.loader import load_level
    from eyesofpriestess.schema import load_clicks

    level = load_level(GAMEDATA / "levels" / "obt" / "main" / f"level_main_{args.stage}.json")
    print(f"关卡 {level.level_id} 出怪 {len(level.spawns)} 次")

    # 一次性推演完整时长，后面按时间查帧（比每个候选时刻重跑快几百倍）
    result = simulate(level, None, duration=args.t_max, fps=2.0, dt=0.1)
    print(f"推演完成：{len(result.frames)} 帧，时长 {result.duration:.1f}s")

    clicks = load_clicks(Path(args.clicks))
    items = clicks[args.frame]
    dst = np.array([[c.x, c.y] for c in items if not c.is_ignore], dtype=np.float64)
    print(f"{args.frame}: {len(dst)} 个点")
    if len(dst) < 4:
        print("点太少，换一帧（同一帧至少 4 个敌人）")
        return 1

    best = None
    time_s = 1.0
    while time_s <= args.t_max:
        frame = result.frame_at(time_s)
        src = np.array([[e.col, e.row] for e in frame.enemies], dtype=np.float64)
        if len(src) >= 4 and abs(len(src) - len(dst)) <= max(3, len(dst)):
            fit = icp_fit(src, dst)
            if fit is not None:
                matrix, residual, inliers = fit
                if inliers >= 4 and residual < 80:
                    key = (inliers, -residual)
                    if best is None or key > (best[1], -best[2]):
                        best = (matrix, inliers, residual, time_s, len(src))
        time_s += args.step

    if best is None:
        print("没对上：确认 --stage 和帧所属关卡一致，或放宽 --t-max")
        return 1
    matrix, inliers, residual, time_s, count = best
    print(f"最佳: 推演 {time_s:.1f}s | 敌人 {count} | 内点 {inliers}/{len(dst)} | 残差 {residual:.1f}px")
    print("单应矩阵 (col,row) -> (x,y):")
    for row in matrix:
        print("  " + " ".join(f"{value:11.4f}" for value in row))
    out = ROOT / "out" / f"sim_align_{args.stage}.json"
    out.write_text(json.dumps({"stage": args.stage, "frame": args.frame, "time": time_s,
                               "inliers": inliers, "residual": residual,
                               "homography": matrix.tolist()}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("已写", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
