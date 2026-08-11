"""从 prompts/ 目录加载 Markdown / TXT 提示词模板。"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """按文件名加载提示词（可省略扩展名，优先 .md 再 .txt）。"""
    stem = name[:-3] if name.endswith(".md") else name
    stem = stem[:-4] if stem.endswith(".txt") else stem

    candidates = [
        PROMPTS_DIR / f"{stem}.md",
        PROMPTS_DIR / f"{stem}.txt",
        PROMPTS_DIR / name,
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() + "\n"
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到提示词文件: {name}（尝试过: {tried}）")


def render_prompt(name: str, **kwargs: Any) -> str:
    """加载模板并用 kwargs 做 {{key}} 替换。

    categories 若为 list/tuple，自动序列化为 JSON 数组字符串。
    """
    text = load_prompt(name)
    for key, value in kwargs.items():
        if key == "categories" and isinstance(value, (list, tuple)):
            value = json.dumps(list(value), ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key + "}}", value)
    return text
