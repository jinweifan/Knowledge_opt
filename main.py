#!/usr/bin/env python3
"""项目主入口：客服回复幻觉检测（RAG）。

用法:
  uv run python main.py --top-k 3
  uv run python -m cli.build_index
"""

from __future__ import annotations

from cli.detect import main

if __name__ == "__main__":
    main()
