#!/usr/bin/env python3
"""一键构建 Chroma 索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from kb.paths import DATA_DIR
from kb.pipeline.index import DEFAULT_PERSIST, build_chroma_index


def main() -> None:
    """CLI：从 replies.json 重建 Chroma 知识库索引。"""
    parser = argparse.ArgumentParser(description="构建客服知识库 Chroma 索引")
    parser.add_argument(
        "--replies",
        type=Path,
        default=DATA_DIR / "replies.json",
    )
    parser.add_argument("--persist", type=Path, default=DEFAULT_PERSIST)
    args = parser.parse_args()

    vs = build_chroma_index(args.replies, args.persist, reset=True)
    n = vs._collection.count()  # noqa: SLF001
    print(f"[ok] Chroma 索引已写入 {args.persist}，文档数={n}")


if __name__ == "__main__":
    main()
