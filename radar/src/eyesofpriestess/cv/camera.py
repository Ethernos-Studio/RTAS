"""战斗相机模型（结论来自群友逆向资料的"结论"段，未使用其任何代码/工具）。

来源与验证（2026-11-14 深夜，作者第二次修正）：

* 矩阵现在是作者修正后的值（V 的平移分量 0.3794 / -9.5028，早前一版是 0.286 / -9.125，
  那版 Y 方向有系统性偏差，作者本人确认"只在一张图上验证过"）；
* 用作者的验证点复核：世界 (0.27, 0.48) -> (1434.7, 496.5)，文档值 (1435, 495)，Δ≈1.5px；
  世界 (1.8, 0.012) -> (1718.9, 567.9)，文档值 (1732, 583)，Δ≈20px——这属于作者列出的
  "脚底原点 / 移动插值"容差（允许 50~100px）；
* 仍然**只是单套参数**：不同地图/关卡请用 ``fit_pose()`` 从自己的对应点重解，
  再用残差判断这套参数是否适用。

资料给出的实测矩阵（Unity 内存里是**列主序**，这里换成数学上的行主序）：

视图矩阵 V（世界 -> 相机）::

    [1,     0,     0,      0    ]
    [0,     0.866, -0.5,   0.286]
    [0,    -0.5,   -0.866, -9.125]
    [0,     0,      0,     1    ]

* 3×3 部分是加俯仰 30° 的旋转（cos30=0.866、sin30=0.5）
* 平移 (0, 0.286, -9.125) —— 相机在地面上方 9.125 个单位

投影矩阵 P（透视，Unity 约定）::

    [1.2687, 0,       0,       0     ]
    [0,      2.7475,  0,       0     ]
    [0,      0,      -1.0006, -0.6002]
    [0,      0,      -1,       0     ]

用资料里的验证点（2772×1280 屏）：world(0,0,0) -> (1386, 585)；
world(1,0,0) -> (1578.7, 585)：即 1 个世界单位 ≈ 193px（横向）。

地图坐标 -> 世界坐标（资料实测，**每张图的 offset 不同**）：:

    world_x = a - map_x
    world_z = map_y - b
    world_y = 0

资料里 TR-4 的 (a, b) = (1.2, 3.5)，换图即失效，所以这里只提供拟合函数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: 资料实测的视图矩阵（已转成行主序）
DEFAULT_VIEW = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.8660254, -0.5, 0.3794],
    [0.0, -0.5, -0.8660254, -9.5028],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

#: 资料实测的投影矩阵（已转成行主序）
DEFAULT_PROJECTION = np.array([
    [1.2687, 0.0, 0.0, 0.0],
    [0.0, 2.7475, 0.0, 0.0],
    [0.0, 0.0, -1.0006, -0.6002],
    [0.0, 0.0, -1.0, 0.0],
], dtype=np.float64)


@dataclass
class BattleCamera:
    """战场透视相机：默认用实测矩阵，也可以按 fov/俯仰构造。"""

    screen_w: int = 1920
    screen_h: int = 1080
    view: np.ndarray = field(default_factory=lambda: DEFAULT_VIEW.copy())
    projection: np.ndarray = field(default_factory=lambda: DEFAULT_PROJECTION.copy())

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_pose(
        cls,
        *,
        screen_w: int = 1920,
        screen_h: int = 1080,
        fov_deg: float = 40.0,
        pitch_deg: float = 30.0,
        height: float = 9.5028,
        y_offset: float = 0.3794,
        roll_deg: float = 0.0,
        near: float = 0.6,
        far: float = 100.0,
    ) -> "BattleCamera":
        """按"fov + 俯仰 + 高度"构造——合成数据渲染时改角度用这个。"""
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        tilt = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, -sin_p, -cos_p],
        ], dtype=np.float64)
        roll_matrix = np.array([
            [cos_r, -sin_r, 0.0],
            [sin_r, cos_r, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = roll_matrix @ tilt
        view[1, 3] = y_offset
        view[2, 3] = -height
        aspect = screen_w / screen_h
        focal = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        projection = np.array([
            [focal / aspect, 0.0, 0.0, 0.0],
            [0.0, focal, 0.0, 0.0],
            [0.0, 0.0, -1.0006, -near * 1.0006],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float64)
        return cls(screen_w=screen_w, screen_h=screen_h, view=view, projection=projection)

    # -- 投影 -------------------------------------------------------------
    def project(self, world: tuple[float, float, float]) -> tuple[float, float, float] | None:
        """世界坐标 -> (屏幕 x, 屏幕 y, 深度)；在相机后方返回 None。"""
        point = np.array([world[0], world[1], world[2], 1.0], dtype=np.float64)
        clip = self.projection @ (self.view @ point)
        if abs(clip[3]) < 1e-9:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        x = (ndc[0] + 1.0) * 0.5 * self.screen_w
        y = (1.0 - ndc[1]) * 0.5 * self.screen_h
        return float(x), float(y), float(clip[3])

    def project_map(self, map_x: float, map_y: float, offset: tuple[float, float]) -> tuple[float, float] | None:
        """地图坐标 -> 屏幕像素；offset 是每关不同的 (a, b)。"""
        a, b = offset
        result = self.project((a - map_x, 0.0, map_y - b))
        return None if result is None else (result[0], result[1])


    # -- 位姿拟合（不信任任何预置参数时用这个）----------------------------
    @classmethod
    def fit_pose(
        cls,
        correspondences: list[tuple[float, float, float, float]],
        *,
        screen_w: int = 1920,
        screen_h: int = 1080,
        init: dict | None = None,
    ) -> tuple["BattleCamera", dict]:
        """用「格子坐标 <-> 屏幕像素」对应点拟合 (fov, 俯仰, 高度, 横滚, a, b)。

        坐标下降 + 步长减半，不依赖 SciPy。返回 (相机, 参数与残差报告)。
        需要至少 6 组点、且点不能共线。
        """
        import math

        if len(correspondences) < 6:
            raise ValueError("至少需要 6 组对应点才拟合得出 6 个参数")

        def project_with(params: dict, col: float, row: float):
            camera = cls.from_pose(
                screen_w=screen_w, screen_h=screen_h,
                fov_deg=params["fov"], pitch_deg=params["pitch"],
                height=params["height"], roll_deg=params["roll"],
            )
            return camera.project_map(col, row, (params["a"], params["b"]))

        def loss(params: dict) -> float:
            total = 0.0
            for col, row, x, y in correspondences:
                projected = project_with(params, col, row)
                if projected is None:
                    return float("inf")
                total += (projected[0] - x) ** 2 + (projected[1] - y) ** 2
            return total

        params = {"fov": 40.0, "pitch": 30.0, "height": 9.0, "roll": 0.0, "a": 1.2, "b": 3.5}
        if init:
            params.update({key: float(value) for key, value in init.items()})

        steps = {"fov": 8.0, "pitch": 8.0, "height": 3.0, "roll": 4.0, "a": 3.0, "b": 3.0}
        best = loss(params)
        for _ in range(40):
            improved = True
            while improved:
                improved = False
                for key, step in steps.items():
                    for delta in (step, -step):
                        candidate = dict(params)
                        candidate[key] = params[key] + delta
                        if key in ("fov", "height") and candidate[key] <= 1.0:
                            continue
                        value = loss(candidate)
                        if value < best - 1e-9:
                            params, best = candidate, value
                            improved = True
            steps = {key: value / 2.0 for key, value in steps.items()}
            if max(steps.values()) < 1e-3:
                break

        residuals = []
        for col, row, x, y in correspondences:
            projected = project_with(params, col, row)
            residuals.append(float("inf") if projected is None else
                             ((projected[0] - x) ** 2 + (projected[1] - y) ** 2) ** 0.5)

        camera = cls.from_pose(
            screen_w=screen_w, screen_h=screen_h,
            fov_deg=params["fov"], pitch_deg=params["pitch"],
            height=params["height"], roll_deg=params["roll"],
        )
        report = {
            "params": {key: round(value, 4) for key, value in params.items()},
            "residual_mean_px": round(sum(residuals) / len(residuals), 2),
            "residual_max_px": round(max(residuals), 2),
            "points": len(correspondences),
        }
        return camera, report

    # -- 标定 -------------------------------------------------------------
    def fit_offset(self, correspondences: list[tuple[float, float, float, float]],
                   guess: tuple[float, float] = (1.2, 3.5)) -> tuple[float, float]:
        """用 (map_x, map_y, screen_x, screen_y) 对应点拟合每关的 (a, b)。

        没有 SciPy 也能跑：先粗网格搜，再局部细化。
        """
        def loss(a: float, b: float) -> float:
            total = 0.0
            for map_x, map_y, screen_x, screen_y in correspondences:
                projected = self.project_map(map_x, map_y, (a, b))
                if projected is None:
                    return float("inf")
                total += (projected[0] - screen_x) ** 2 + (projected[1] - screen_y) ** 2
            return total

        best = (guess[0], guess[1], loss(*guess))
        step = 2.0
        for _ in range(6):
            improved = True
            while improved:
                improved = False
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    candidate = (best[0] + dx, best[1] + dy)
                    value = loss(*candidate)
                    if value < best[2] - 1e-6:
                        best = (candidate[0], candidate[1], value)
                        improved = True
            step /= 2.0
        return best[0], best[1]
