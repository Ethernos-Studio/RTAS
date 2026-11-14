"""战斗相机模型（结论来自群友逆向资料的"结论"段，未使用其任何代码/工具）。

警告（2026-11-14 晚更正）：下面的默认矩阵来自**单张地图**的实测，
作者本人确认"只在一张图上验证过、Y 方向可能不对、正在修"。
所以在拿到多尺寸地图的交叉验证之前：

* **不要**用默认矩阵做生产环境的格子坐标换算或合成数据渲染；
* 生产路径用**每关自己标定的单应**（``cv/calib.py``）+ 自己的对应点拟合；
* 本模块保留是因为"投影链公式"（clip/ndc/screen 换算）和"地图→世界是平移关系、
  offset 每关不同"这两条结论是可靠的，且单元测试锁住的只是"文档里那三个验证点"。

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
    [0.0, 0.8660254, -0.5, 0.286],
    [0.0, -0.5, -0.8660254, -9.125],
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
        height: float = 9.125,
        y_offset: float = 0.286,
        near: float = 0.6,
        far: float = 100.0,
    ) -> "BattleCamera":
        """按"fov + 俯仰 + 高度"构造——合成数据渲染时改角度用这个。"""
        pitch = math.radians(pitch_deg)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        view = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p, y_offset],
            [0.0, -sin_p, -cos_p, -height],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
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
