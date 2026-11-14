"""实战止损：叠加层默认置信度提到 0.45，并把身份命中 ally 编号的框压掉。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    path = ROOT / "tools" / "live_overlay.py"
    text = path.read_text(encoding="utf-8")

    old_default = '    parser.add_argument("--conf", type=float, default=0.3)'
    new_default = (
        '    parser.add_argument("--conf", type=float, default=0.45,\n'
        '                   help="置信度阈值（实战 0.45~0.55：太低会把干员当成敌人）")'
    )
    if old_default in text:
        text = text.replace(old_default, new_default)
        print("conf 默认值已提高")
    else:
        print("conf 参数没匹配到")

    marker = "        self.model = YOLO(str(config.weights))"
    if "ally_codes" not in text and marker in text:
        text = text.replace(
            marker,
            marker + "\n"
            "        # 身份命中 ally 编号（U17 之类）的框直接压掉：\n"
            "        # 干员被误检成敌人是实战里最大的噪声源\n"
            "        self.ally_codes = {code for code, label in _load_catalog_codes().items()\n"
            "                           if label == 'ally'}",
            1,
        )
        print("ally_codes 已加")

    helper = '''

def _load_catalog_codes() -> dict[str, str]:
    """读类别表：编号 -> 粗类（enemy/ally）。"""
    import json

    path = ROOT / "configs" / "enemies.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    codes: dict[str, str] = {}
    for code, item in (data.get("enemies") or {}).items():
        codes[str(code)] = str(item.get("cls") or "enemy")
    for code, item in (data.get("special") or {}).items():
        codes[str(code)] = str(item.get("cls") or "enemy")
    return codes
'''
    if "def _load_catalog_codes" not in text:
        text = text.replace("\ndef _latest_weights() -> Path:", helper + "\n\ndef _latest_weights() -> Path:", 1)
        print("类别表读取函数已加")

    ast.parse(text)
    path.write_text(text, encoding="utf-8")
    print("语法 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
