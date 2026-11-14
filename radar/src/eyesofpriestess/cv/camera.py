"""战斗相机模型（结论来自群友逆向资料的结论段，未使用其任何代码/工具）。

采用的结论：
* 战斗相机是**透视**：fov 40°，位置约 (0, -0.30, 9.11)，俯仰 30°；
  剧情/AVG 用的 `AVGCamera_Scene` 才是正交 3.6，战场不用它。
* 投影链：`clip = M·(x,y,z,1)`（列主序），`ndc = clip.xy / clip.w`，
  `screen = ((ndc.x+1)/2·W, (1-ndc.y)/2·H)`。
* 本作 Transform 层级里旋转恒为单位、scale 恒为 1 → **世界坐标 = 沿父链把 position 直接相加**。
* 地图坐标→世界坐标：`world_x = a - map_x`、`world_z = map_y - b`、`world_y = 0`；
  但 **a/b 每张图不同**（资料里 TR-4 的 1.2 / 3.5 换图就失效），必须每关标一次。
* 投影误差：y 方向约偏 25~30px（模型脚底与 Transform 原点之差）——用合成标签时记得补这个偏移。

这些只是"通过图像验证过的相机参数"，与反作弊/内存读取无关；
本模块把它们写成纯数学，供合成数据渲染和格子坐标换算使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BattleCamera:
    """战场透视相机（默认参数来自实测结论）。"""

    screen_w: int = 1920
    screen_h: int = 1080
    fov_deg: float = 40.0
    position: tuple[float, float, float] = (0.0, -0.30, 9.11)
    pitch_deg: float = 30.0          # 相机向下俯视角度

    def _view(self) -> np.ndarray:
        """世界 -> 相机（列主序风格：world2cam）。"""
        pitch = math.radians(self.pitch_deg)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        # 先绕 X 轴俯仰（让相机朝下看地面），再平移到相机位置
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p],
        ], dtype=np.float64)
        camera = np.eye(4, dtype=np.float64)
        camera[:3, :3] = rotation
        camera[:3, 3] = -rotation @ np.array(self.position, dtype=np.float64)
        return camera

    def _projection(self) -> np.ndarray:
        fov = math.radians(self.fov_deg)
        aspect = self.screen_w / self.screen_h
        focal = 1.0 / math.tan(fov / 2.0)
        projection = np.zeros((4, 4), dtype=np.float64)
        projection[0, 0] = focal / aspect
        projection[1, 1] = focal
        projection[2, 2] = -1.002
        projection[2, 3] = -1.0
        projection[3, 2] = -0.06
        return projection

    def project(self, world: tuple[float, float, float]) -> tuple[float, float, float] | None:
        """世界坐标 -> (屏幕 x, 屏幕 y, 深度)；在相机后方返回 None。"""
        point = np.array([*world, 1.0], dtype=np.float64)
        clip = self._projection() @ (self._view() @ point)
        if abs(clip[3]) < 1e-9:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] <= 0:
            return None
        x = (ndc[0] + 1.0) * 0.5 * self.screen_w
        y = (1.0 - ndc[1]) * 0.5 * self.screen_h
        return float(x), float(y), float(clip[3])

    def project_map(self, map_x: float, map_y: float, offset: tuple[float, float]) -> tuple[float, float] | None:
        """地图格子坐标（浮点）-> 屏幕像素。offset 是每关不同的 (a, b)。"""
        a, b = offset
        world = (a - map_x, 0.0, map_y - b)
        result = self.project(world)
        return None if result is None else (result[0], result[1])


def fit_offset(
    camera: BattleCamera,
    correspondences: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """用 (map_x, map_y, screen_x, screen_y) 对应点最小二乘拟合每关的 (a, b)。

    相机固定、只差平移偏移时，用 Nelder-Mead 在 (a,b) 上搜最小重投影误差即可。
    """
    from scipy.optimize import minimize

    def loss(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        total = 0.0
        for map_x, map_y, screen_x, screen_y in correspondences:
            projected = camera.project_map(map_x, map_y, (a, b))
            if projected is None:
                return 1e9
            total += (projected[0] - screen_x) ** 2 + (projected[1] - screen_y) ** 2
        return total

    result = minimize(loss, np.array([1.2, 3.5]), method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])
