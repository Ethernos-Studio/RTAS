"""把场景图翻译成"VLM 能读的文字"。

VLM 看不了坐标——给它 ``{"box": [x, y, w, h]}`` 等于没给。所以这一层把像素位置
翻成**方位 + 数量 + 身份**，让模型能直接拿去做判断：

    t=3.1s：敌人 4（B2×3、未识别×1）；左上 2、右下 1、中上 1
    t=4.1s：敌人 5（新增 1，B2）；左侧 3、右下 2

方位用画面九宫格，不依赖关卡映射；以后 RTAS 给了格子坐标，可以把 ``zone`` 换成格号。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

#: 九宫格的文字名（行, 列）
ZONE_NAMES = {
    (0, 0): "左上", (0, 1): "中上", (0, 2): "右上",
    (1, 0): "左侧", (1, 1): "中央", (1, 2): "右侧",
    (2, 0): "左下", (2, 1): "中下", (2, 2): "右下",
}


def zone_of(center_x: float, center_y: float, width: int, height: int) -> str:
    """像素中心点 -> 九宫格方位名。"""
    col = 0 if center_x < width / 3 else (1 if center_x < width * 2 / 3 else 2)
    row = 0 if center_y < height / 3 else (1 if center_y < height * 2 / 3 else 2)
    return ZONE_NAMES[(row, col)]


@dataclass
class Describer:
    """把逐帧 units 汇总成一句话。"""

    width: int = 1920
    height: int = 1080
    unknown: str = "未识别"
    names: dict[str, str] = field(default_factory=dict)   # 编号 -> 中文名（VLM 读名字更省 token）

    def label_for(self, code: str) -> str:
        name = self.names.get(code)
        return f"{code} {name}" if name else code

    def summarize(self, units: list[dict[str, Any]]) -> dict[str, Any]:
        enemies = [unit for unit in units if unit.get("cls") == "enemy"]
        others = [unit for unit in units if unit.get("cls") != "enemy"]
        codes: dict[str, int] = {}
        zones: dict[str, int] = {}
        for unit in enemies:
            x, y, w, h = unit["box"]
            code = unit.get("code") or self.unknown
            codes[code] = codes.get(code, 0) + 1
            zone = zone_of(x + w / 2, y + h / 2, self.width, self.height)
            zones[zone] = zones.get(zone, 0) + 1
        return {"enemies": len(enemies), "codes": codes, "zones": zones,
                "others": len(others)}

    def line(self, timestamp: float, summary: dict[str, Any]) -> str:
        if not summary["enemies"]:
            return f"t={timestamp:.1f}s：画面内没有敌人"
        codes = "、".join(f"{self.label_for(code)}×{count}" for code, count in
                          sorted(summary["codes"].items(), key=lambda kv: -kv[1]))
        zones = "、".join(f"{zone} {count}" for zone, count in
                          sorted(summary["zones"].items(), key=lambda kv: -kv[1]))
        return f"t={timestamp:.1f}s：敌人 {summary['enemies']}（{codes}）；{zones}"

    def unit_lines(self, units: list[dict[str, Any]]) -> list[str]:
        """逐个敌人给一行：编号 + 置信度 + **中心坐标** + 尺寸 + 方位。

        VLM 拿到图 + 这段文字，才能把"图里那个家伙"和"坐标"对上；
        只给"左上 2 个"它是没法指认具体哪一个的。
        """
        lines = []
        for unit in units:
            if unit.get("cls") != "enemy":
                continue
            x, y, w, h = unit["box"]
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            zone = zone_of(center_x, center_y, self.width, self.height)
            code = unit.get("code") or self.unknown
            score = unit.get("code_score")
            code_text = self.label_for(code) + (f"({score:.2f})" if isinstance(score, (int, float)) else "")
            track = unit.get("track_id")
            lines.append(
                f"  #{track} {code_text} conf={unit.get('conf', 0):.2f} "
                f"center=({center_x},{center_y}) size={w}x{h} {zone}"
            )
        return lines


def describe(
    records: Iterable[dict[str, Any]],
    *,
    width: int = 1920,
    height: int = 1080,
    every: float = 1.0,
) -> list[str]:
    """每 ``every`` 秒输出一条描述。

    逐帧描述会"2→4→2→3"地抖（检测本身在抖），VLM 读起来会以为敌人在反复刷新。
    所以按时间窗聚合：取窗口内**帧计数的中位数**，再挑一个接近中位数的帧来描述。
    """
    describer = Describer(width=width, height=height)
    bucket: list[dict[str, Any]] = []
    bucket_end = every

    def flush(items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return []
        counts = sorted(len([u for u in item.get("units") or [] if u.get("cls") == "enemy"])
                        for item in items)
        median = counts[len(counts) // 2]
        chosen = min(items, key=lambda item: abs(
            len([u for u in item.get("units") or [] if u.get("cls") == "enemy"]) - median))
        units = chosen.get("units") or []
        summary = describer.summarize(units)
        return [describer.line(float(chosen.get("t", 0.0)), summary)] + describer.unit_lines(units)

    output: list[str] = []
    for record in records:
        timestamp = float(record.get("t", 0.0))
        if timestamp > bucket_end:
            output.extend(flush(bucket))
            bucket = []
            while timestamp > bucket_end:
                bucket_end += every
        bucket.append(record)

    output.extend(flush(bucket))
    return output


def load_scene(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
