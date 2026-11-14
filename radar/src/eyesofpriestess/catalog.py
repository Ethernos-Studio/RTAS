"""标注标签目录：图鉴编号 -> YOLO 粗类。

一笔标注的 ``cls`` 存的是**图鉴编号**（``B1`` / ``A3`` / ``CS`` …），
而 YOLO 只认 2 个粗类（``enemy`` / ``ally``，见 HANDOFF §2.7）。
血条**不进检测类别**：它是半透明黑底 + 纯色填充，用颜色阈值 CV 就能稳定拿到。
``tile``（路障/装置）也不进：**可交互物品的位置在游戏的地图数据里有**，
模型没必要学，标它只是给数据集加噪声。

这一层负责把标签和粗类对上，并提供 Label Studio 标注界面的生成器：

    uv run radar ls-interface      # configs/enemies.json -> labelstudio/interface.xml

编号从哪来：游戏「档案 -> 敌人图鉴」里每个敌人左上角的那个码（B1、A3、D1、19…）。
它跨语言、跨版本稳定，比"源石虫·α"这种中文名可靠得多；中文名只做注释用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

#: Label Studio 里 ignore 标签的前缀（转换器据此写 type:"ignore"）
IGNORE_PREFIX = "IGNORE-"

DEFAULT_CATALOG = Path("configs/enemies.json")

#: 图鉴没收录 / 人手滑打错的编号落到这里
FALLBACK_LABEL = "other_enemy"

#: Label Studio 里各粗类的颜色（ignore 单独一色，抢眼）
CLASS_COLORS = {
    "ally": "#a6f",
    "ignore": "#f44",
}


@dataclass(frozen=True)
class Label:
    """一个标注标签。"""

    value: str                    # LS 里显示的文本 = 图鉴编号
    cls: str                      # YOLO 粗类（enemy / operator / …）
    hotkey: str = ""
    name: str = ""                # 中文名，仅注释
    tier: str = "normal"          # normal / elite / leader

    @property
    def is_ignore(self) -> bool:
        return self.value.startswith(IGNORE_PREFIX)

    @property
    def ignore_reason(self) -> str:
        return self.value[len(IGNORE_PREFIX):]


class Catalog:
    """标签目录。加载后即可回答「这个标签该落哪个 YOLO 类」。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.classes: list[str] = list(data.get("classes") or [])
        self.labels: dict[str, Label] = {}
        for code, item in (data.get("enemies") or {}).items():
            item = item or {}
            self.labels[code] = Label(
                value=str(code),
                cls=str(item.get("cls") or "enemy"),
                hotkey=str(item.get("hotkey") or ""),
                name=str(item.get("name") or ""),
                tier=str(item.get("tier") or "normal"),
            )
        for value, item in (data.get("special") or {}).items():
            item = item or {}
            self.labels[str(value)] = Label(
                value=str(value),
                cls=str(item.get("cls") or "enemy"),
                hotkey=str(item.get("hotkey") or ""),
            )
        for value, item in (data.get("ignore") or {}).items():
            item = item or {}
            self.labels[str(value)] = Label(
                value=str(value),
                cls="ignore",
                hotkey=str(item.get("hotkey") or ""),
            )
        if not self.classes:
            raise ValueError("catalog 缺少 classes")

    # -- 查询 -------------------------------------------------------------
    def label(self, value: str) -> Label | None:
        return self.labels.get(value)

    def cls_for(self, value: str) -> str | None:
        """标签 -> YOLO 类名；``None`` 表示这是 ignore 标签。"""
        label = self.labels.get(value)
        if label is None:
            return None if value.startswith(IGNORE_PREFIX) else None
        if label.is_ignore:
            return None
        if label.cls not in self.classes:
            raise ValueError(
                f"标签 {value!r} 映射到未知类别 {label.cls!r}；"
                f"要么补进 catalog.classes，要么改它的 cls"
            )
        return label.cls

    def unknown_label(self, value: str) -> bool:
        return value not in self.labels and not value.startswith(IGNORE_PREFIX)

    def enemy_codes(self) -> list[str]:
        return [label.value for label in self.labels.values() if label.cls == "enemy"]

    # -- 生成 LS 界面 -----------------------------------------------------
    def interface_xml(self, title: str = "EYESOFPRIESTESS · 点选") -> str:
        """生成 Label Studio 的 ``KeyPointLabels`` 配置。

        顺序 = ``configs/enemies.json`` 里的声明顺序（按常用度排的），
        所以标注时视线从上往下扫、热键也按这个顺序记。
        """
        lines = [
            "<!-- 由 `uv run radar ls-interface` 从 configs/enemies.json 生成，别手改 -->",
            f"<!-- {title}：先按热键选标签，再点目标；ignore 只点坐标，不出正样本 -->",
            "<View>",
            '  <Image name="img" value="$image" zoom="true" zoomControl="true" '
            'rotateControl="false"/>',
            '  <KeyPointLabels name="pts" toName="img" strokeWidth="2">',
        ]
        for label in self.labels.values():
            if label.cls == "enemy":
                background = {"normal": "#2a6", "elite": "#6af", "leader": "#fa4"}.get(
                    label.tier, "#888"
                )
                suffix = f"  <!-- {label.name} -->" if label.name else ""
                lines.append(self._label_line(label, background) + suffix)
        # 其余 4 个粗类（ally / tile / hp_bar_*）：有标签就配上
        for label in self.labels.values():
            if label.cls in ("enemy", "ignore"):
                continue
            lines.append(self._label_line(label, CLASS_COLORS.get(label.cls, "#888")))
        for label in self.labels.values():
            if label.is_ignore:
                lines.append(self._label_line(label, CLASS_COLORS["ignore"]))
        lines += ["  </KeyPointLabels>", "</View>", ""]
        return "\n".join(lines)

    @staticmethod
    def _label_line(label: Label, background: str) -> str:
        hotkey = f' hotkey="{label.hotkey}"' if label.hotkey else ""
        return (
            f'    <Label value="{label.value}"{hotkey} '
            f'background="{background}" showAlias="false"/>'
        )

    def cheat_sheet(self) -> str:
        """人看的键位表（标注时放旁边）。"""
        rows = ["键位  标签        类别        中文名", "-" * 48]
        for label in self.labels.values():
            if not label.hotkey:
                continue
            cls = label.ignore_reason if label.is_ignore else label.cls
            rows.append(f"{label.hotkey:<5} {label.value:<11} {cls:<10} {label.name}")
        return "\n".join(rows)


def load_catalog(path: Path | str | None = None) -> Catalog:
    """读 ``configs/enemies.json``；路径按项目根解析。"""
    from .paths import find_root  # 局部导入，避免循环

    if path is None:
        path = find_root() / DEFAULT_CATALOG
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到标签目录: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Catalog(data)


def resolve_labels(catalog: Catalog, labels: Iterable[str]) -> list[str]:
    """把一批标签翻成 YOLO 类名（ignore 跳过），供 dataset 用。"""
    out: list[str] = []
    for value in labels:
        cls = catalog.cls_for(value)
        if cls:
            out.append(cls)
    return out
