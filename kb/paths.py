"""项目根路径与常用目录。"""

from __future__ import annotations

from pathlib import Path

# kb/paths.py → 仓库根目录
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
INDEX_DIR = ROOT / "index" / "chroma_kb"
