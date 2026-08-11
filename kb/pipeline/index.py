"""Chroma 知识库索引：从 replies 中的 knowledge_base 建库。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from kb.paths import INDEX_DIR
from kb.pipeline.llm import get_embeddings

DEFAULT_PERSIST = INDEX_DIR
COLLECTION = "customer_service_kb"


def load_replies(path: Path) -> list[dict[str, Any]]:
    """从 JSON 文件加载 replies 列表。

    Args:
        path: replies.json 路径。

    Returns:
        回复样本字典列表。
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def replies_to_documents(replies: list[dict[str, Any]]) -> list[Document]:
    """将 replies 中的 knowledge_base 转为 LangChain Document。

    Args:
        replies: 含 id / knowledge_base 等字段的样本。

    Returns:
        可写入向量库的 Document 列表。
    """
    docs: list[Document] = []
    for item in replies:
        text = (item.get("knowledge_base") or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": item["id"],
                    "source": "replies.knowledge_base",
                    "user_question": item.get("user_question", "")[:200],
                },
            )
        )
    return docs


def build_chroma_index(
    replies_path: Path,
    persist_dir: Path = DEFAULT_PERSIST,
    *,
    reset: bool = True,
) -> Chroma:
    """构建并持久化 Chroma 索引。

    Args:
        replies_path: replies.json 路径。
        persist_dir: Chroma 持久化目录。
        reset: True 时先清空目录再重建。

    Returns:
        已写入文档的 Chroma 向量库。

    Raises:
        ValueError: 知识库文档为空。
    """
    # 每条 reply 的 knowledge_base 作为一条可检索文档
    replies = load_replies(replies_path)
    docs = replies_to_documents(replies)
    if not docs:
        raise ValueError("知识库文档为空，无法建索引")

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    if reset and persist_dir.exists():
        # 重建时清空目录，避免旧 collection 残留导致脏命中
        shutil.rmtree(persist_dir, ignore_errors=True)
        persist_dir.mkdir(parents=True, exist_ok=True)

    # 嵌入后持久化到本地，检测阶段直接 load
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
    )


def load_chroma_index(persist_dir: Path = DEFAULT_PERSIST) -> Chroma:
    """加载已持久化的 Chroma 索引。

    Args:
        persist_dir: Chroma 持久化目录。

    Returns:
        可检索的 Chroma 实例。

    Raises:
        FileNotFoundError: 索引目录不存在。
    """
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"未找到索引目录 {persist_dir}，请先运行: python -m cli.build_index"
        )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def retrieve_knowledge(
    vectorstore: Chroma,
    query: str,
    *,
    top_k: int = 3,
    exclude_doc_id: str | None = None,
) -> list[Document]:
    """向量检索知识片段。

    Args:
        vectorstore: Chroma 向量库。
        query: 检索查询文本。
        top_k: 返回条数。
        exclude_doc_id: 排除本条对应知识，避免泄漏。

    Returns:
        命中的 Document 列表（metadata 含 retrieve_score）。
    """
    # 排除自身时多取若干条再过滤，保证凑满 top_k
    fetch_k = top_k + (3 if exclude_doc_id else 0)
    pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
    results: list[Document] = []
    for doc, score in pairs:
        doc.metadata["retrieve_score"] = float(score)
        if exclude_doc_id and doc.metadata.get("doc_id") == exclude_doc_id:
            continue
        results.append(doc)
        if len(results) >= top_k:
            break
    return results
