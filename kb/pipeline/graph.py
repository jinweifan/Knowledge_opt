"""LangGraph：检索 → DeepSeek 幻觉判定。"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from kb.pipeline.index import load_chroma_index, retrieve_knowledge
from kb.pipeline.llm import get_deepseek_chat
from kb.prompts import render_prompt
from kb.taxonomy import CATEGORY_NAMES, TAXONOMY


class DetectState(TypedDict, total=False):
    """LangGraph 检测流水线状态（检索 → 判定各节点读写）。

    Attributes:
        id: 样本编号。
        user_question: 用户原问题。
        system_reply: 待检测的客服回复。
        gold_knowledge: 样本自带的金标知识（评估用，判定不直接采信）。
        top_k: 向量检索返回条数。
        exclude_self: 检索时是否排除本条金标文档，避免泄漏。
        query: 实际用于检索的查询文本。
        retrieved_docs: 检索命中明细（含 doc_id / score / content）。
        retrieved_contexts: 检索到的知识正文列表，供判定模型作唯一事实源。
        is_hallucination: 是否判定为幻觉。
        hallucination_type: 幻觉类型名；无幻觉时为 None。
        severity: 严重程度（critical/high/medium/low/none）。
        confidence: 模型置信度，0～1。
        evidence: 判定依据（通常引用检索上下文）。
        detail: 一两句判定理由。
        mode: 检测模式标识（如 langgraph-chroma-deepseek）。
    """

    id: str
    user_question: str
    system_reply: str
    gold_knowledge: str
    top_k: int
    exclude_self: bool
    query: str
    retrieved_docs: list[dict[str, Any]]
    retrieved_contexts: list[str]
    is_hallucination: bool
    hallucination_type: str | None
    severity: str
    confidence: float
    evidence: str
    detail: str
    mode: str


JUDGE_SYSTEM = render_prompt("hallucination_rag", categories=CATEGORY_NAMES)


def _normalize_type(raw: str | None, is_h: bool) -> str | None:
    """将模型输出的类型名归一到 TAXONOMY 键。"""
    if not is_h:
        return None
    if not raw:
        return "信息编造"
    raw = str(raw).strip()
    if raw in TAXONOMY:
        return raw
    for name in CATEGORY_NAMES:
        if name in raw:
            return name
    return "信息编造"


def _parse_json(content: str) -> dict[str, Any]:
    """解析判定模型返回的 JSON。"""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise
        return json.loads(match.group(0))


def build_query(user_question: str, system_reply: str) -> str:
    """构造检索 query（用户问题 + 回复摘要）。

    Args:
        user_question: 用户问题。
        system_reply: 客服回复全文。

    Returns:
        用于向量检索的查询文本。
    """
    reply_short = system_reply.strip().replace("\n", " ")
    if len(reply_short) > 120:
        reply_short = reply_short[:120]
    return f"问题：{user_question}\n客服回复要点：{reply_short}"


def node_retrieve(state: DetectState) -> DetectState:
    """图节点：向量检索相关知识。"""
    vs = load_chroma_index()
    # 用「问题 + 回复摘要」作 query，比单用问题更贴近幻觉证据
    query = build_query(state["user_question"], state["system_reply"])
    # 消融：排除本条对应知识，模拟「未把答案直接塞进上下文」
    exclude = state["id"] if state.get("exclude_self", False) else None
    docs = retrieve_knowledge(
        vs,
        query,
        top_k=int(state.get("top_k", 3)),
        exclude_doc_id=exclude,
    )
    serialized = [
        {
            "doc_id": d.metadata.get("doc_id"),
            "score": d.metadata.get("retrieve_score"),
            "content": d.page_content,
        }
        for d in docs
    ]
    return {
        **state,
        "query": query,
        "retrieved_docs": serialized,
        "retrieved_contexts": [d.page_content for d in docs],
    }


def node_judge(state: DetectState) -> DetectState:
    """图节点：基于检索上下文调用 DeepSeek 判定幻觉。"""
    llm = get_deepseek_chat(temperature=0.0)
    payload = {
        "id": state["id"],
        "user_question": state["user_question"],
        "system_reply": state["system_reply"],
        "retrieved_contexts": state.get("retrieved_contexts") or [],
    }
    # temperature=0：尽量稳定；仅依据检索上下文，不直喂本条标准答案知识
    msg = llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": "请基于检索上下文检测幻觉，输出 JSON：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    data = _parse_json(content)
    is_h = bool(data.get("is_hallucination"))
    # 类型名归一到 taxonomy，非法/空值回退「信息编造」
    h_type = _normalize_type(data.get("hallucination_type"), is_h)
    severity = data.get("severity") or (
        TAXONOMY[h_type].severity if h_type in TAXONOMY else "medium"
    )
    if not is_h:
        severity = "none"
        h_type = None
    try:
        confidence = float(data.get("confidence", 0.8))
    except (TypeError, ValueError):
        confidence = 0.8
    return {
        **state,
        "is_hallucination": is_h,
        "hallucination_type": h_type,
        "severity": severity,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": str(data.get("evidence") or ""),
        "detail": str(data.get("detail") or ""),
        "mode": "langgraph-chroma-deepseek",
    }


def build_detect_graph():
    """编译「检索 → 判定」LangGraph。

    Returns:
        可 ``invoke`` 的编译图。
    """
    # 固定两节点流水线：retrieve → judge
    g = StateGraph(DetectState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("judge", node_judge)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_edge("judge", END)
    return g.compile()


def detect_one_rag(
    item: dict[str, Any],
    *,
    top_k: int = 3,
    exclude_self: bool = False,
    graph=None,
) -> dict[str, Any]:
    """对单条回复执行 RAG 幻觉检测。

    Args:
        item: 含 id / user_question / system_reply / knowledge_base 的样本。
        top_k: 检索条数。
        exclude_self: 是否排除本条对应知识。
        graph: 可选预编译图，批量时复用。

    Returns:
        含判定字段与 retrieval 详情的结果字典。
    """
    app = graph or build_detect_graph()
    # 跑图：检索上下文 → LLM 输出是否幻觉/类型
    out = app.invoke(
        {
            "id": item["id"],
            "user_question": item["user_question"],
            "system_reply": item["system_reply"],
            "gold_knowledge": item.get("knowledge_base", ""),
            "top_k": top_k,
            "exclude_self": exclude_self,
        }
    )
    retrieved = out.get("retrieved_docs") or []
    gold_id = item["id"]
    # 记录「本条对应知识」是否进 Top-K，供检索质量与消融对照
    gold_hit = any(d.get("doc_id") == gold_id for d in retrieved)
    return {
        "id": item["id"],
        "is_hallucination": bool(out.get("is_hallucination")),
        "hallucination_type": out.get("hallucination_type"),
        "severity": out.get("severity", "none"),
        "confidence": float(out.get("confidence") or 0.0),
        "evidence": out.get("evidence") or "",
        "detail": out.get("detail") or "",
        "mode": out.get("mode") or "langgraph-chroma-deepseek",
        "retrieval": {
            "query": out.get("query"),
            "top_k": top_k,
            "exclude_self": exclude_self,
            "contexts": out.get("retrieved_contexts") or [],
            "docs": retrieved,
            "gold_in_retrieved": gold_hit,
        },
    }


def detect_batch_rag(
    items: list[dict[str, Any]],
    *,
    top_k: int = 3,
    exclude_self: bool = False,
) -> list[dict[str, Any]]:
    """批量 RAG 幻觉检测（复用同一编译图）。

    Args:
        items: 回复样本列表。
        top_k: 检索条数。
        exclude_self: 是否排除本条对应知识。

    Returns:
        与输入顺序一致的结果列表。
    """
    # 批量复用同一编译图，避免每条重复 compile
    graph = build_detect_graph()
    return [
        detect_one_rag(item, top_k=top_k, exclude_self=exclude_self, graph=graph)
        for item in items
    ]
