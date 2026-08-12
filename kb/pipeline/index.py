"""Chroma 知识库索引：从 replies 中的 knowledge_base 建库。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from kb.paths import INDEX_DIR
from kb.pipeline.llm import get_embeddings

DEFAULT_PERSIST = INDEX_DIR
COLLECTION = "customer_service_kb"

# 知识正文含这些表述时，视为「系统能力/接口边界」类文档（与具体业务事实区分）
_CAPABILITY_KB_RE = re.compile(
    r"未接入|不具备|不可口头|需人工|无法查询|不能查询|无（客服系统"
)


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
                    # 供能力类二次检索过滤；规则基于措辞，不绑定样本 id
                    "doc_kind": (
                        "capability"
                        if _CAPABILITY_KB_RE.search(text)
                        else "fact"
                    ),
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
    doc_kind: str | None = None,
) -> list[Document]:
    """向量检索知识片段。

    Args:
        vectorstore: Chroma 向量库。
        query: 检索查询文本。
        top_k: 返回条数。
        exclude_doc_id: 排除本条对应知识，避免泄漏。
        doc_kind: 可选，仅保留 metadata.doc_kind 匹配的文档（如 capability）。

    Returns:
        命中的 Document 列表（metadata 含 retrieve_score）。
    """
    # 排除/过滤时多取若干条再筛，尽量凑满 top_k
    fetch_k = top_k + 5
    pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
    results: list[Document] = []
    for doc, score in pairs:
        doc.metadata["retrieve_score"] = float(score)
        if exclude_doc_id and doc.metadata.get("doc_id") == exclude_doc_id:
            continue
        if doc_kind and doc.metadata.get("doc_kind") != doc_kind:
            continue
        results.append(doc)
        if len(results) >= top_k:
            break
    return results


def merge_retrieved_docs(
    *doc_lists: list[Document],
    top_k: int,
) -> list[Document]:
    """按 doc_id 去重合并多路检索结果，保留先出现者（调用方控制优先级）。"""
    seen: set[str] = set()
    merged: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            doc_id = str(doc.metadata.get("doc_id") or doc.page_content[:32])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(doc)
            if len(merged) >= top_k:
                return merged
    return merged
