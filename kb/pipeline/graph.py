"""LangGraph：检索 → DeepSeek 幻觉判定。"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from kb.pipeline.index import (
    load_chroma_index,
    merge_retrieved_docs,
    retrieve_knowledge,
)
from kb.pipeline.llm import get_deepseek_chat
from kb.pipeline.typing_rules import stabilize_judgment
from kb.prompts import render_prompt
from kb.taxonomy import CATEGORY_NAMES

# 回复/问题中出现「假装已查询或已执行」信号时，加一路能力边界检索
_CAPABILITY_CLAIM_RE = re.compile(
    r"帮您查|我帮您查|查了一下|已经在处理|已帮您|已修改|我已经将|"
    r"已升级|预计明天到账|预计.*到账|目前在.+转运|专属客服.*联系"
)
_CAPABILITY_QUESTION_RE = re.compile(
    r"到哪了|物流|快递|退款|进度|改.*地址|修改.*地址|投诉|工单"
)


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
        claims: 模型抽出的断言与证据标签（可泛化类型决策的中间结果）。
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
    claims: list[dict[str, Any]]


JUDGE_SYSTEM = render_prompt("hallucination_rag", categories=CATEGORY_NAMES)


def needs_capability_retrieval(user_question: str, system_reply: str) -> bool:
    """判断是否应追加「系统能力/接口边界」检索路。"""
    return bool(
        _CAPABILITY_CLAIM_RE.search(system_reply)
        or _CAPABILITY_QUESTION_RE.search(user_question)
    )


def build_capability_query(user_question: str, system_reply: str) -> str:
    """构造能力边界检索 query（与具体样本 id 无关）。"""
    return (
        "客服系统能力 接口是否接入 是否不具备查询或操作 "
        f"问题：{user_question}\n回复：{system_reply[:100]}"
    )


def _message_text(msg: Any) -> str:
    """从 ChatMessage 提取纯文本（兼容空响应、多段 content）。"""
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        content = "".join(parts)
    text = str(content or "").strip()
    if text:
        return text
    # 部分模型把正文放在 additional_kwargs
    extra = getattr(msg, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "output_text"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _parse_json(content: str) -> dict[str, Any]:
    """解析判定模型返回的 JSON（容忍 Markdown 围栏与前后杂质）。"""
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("Expecting value", content or "", 0)
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise json.JSONDecodeError("JSON root must be object", text, 0)
    return data


def _invoke_judge_json(llm: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """调用判定模型并解析 JSON；空响应或坏 JSON 时最多重试 2 次。"""
    # 尽量强制 JSON，减少 flash 模型偶发空输出
    bound = llm
    try:
        bound = llm.bind(response_format={"type": "json_object"})
    except Exception:
        bound = llm

    user_base = (
        "请只做断言抽取与证据标签标注。"
        "只输出一个 JSON 对象，不要 Markdown，不要解释，不要输出 hallucination_type。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    last_err: Exception | None = None
    content = ""
    for attempt in range(3):
        suffix = ""
        if attempt > 0:
            suffix = (
                f"\n\n【重试 {attempt}】上一次返回无法解析为 JSON"
                f"（预览={content[:120]!r}）。请重新只输出合法 JSON 对象。"
            )
        msg = bound.invoke(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_base + suffix},
            ]
        )
        content = _message_text(msg)
        try:
            return _parse_json(content)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(
        f"判定模型连续返回无效 JSON（样本 {payload.get('id')}），"
        f"最后内容预览={content[:200]!r}"
    ) from last_err


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
    """图节点：主路语义检索 + 可选能力边界检索。"""
    vs = load_chroma_index()
    query = build_query(state["user_question"], state["system_reply"])
    exclude = state["id"] if state.get("exclude_self", False) else None
    top_k = int(state.get("top_k", 3))

    primary = retrieve_knowledge(
        vs,
        query,
        top_k=top_k,
        exclude_doc_id=exclude,
    )

    # 检测到「假装执行/查询」话术时，追加能力类文档
    if needs_capability_retrieval(state["user_question"], state["system_reply"]):
        capability_docs = retrieve_knowledge(
            vs,
            build_capability_query(state["user_question"], state["system_reply"]),
            top_k=max(2, top_k - 1),
            exclude_doc_id=exclude,
            doc_kind="capability",
        )
        docs = merge_retrieved_docs(capability_docs, primary, top_k=top_k)
        if not docs:
            docs = primary
    else:
        docs = primary

    serialized = [
        {
            "doc_id": d.metadata.get("doc_id"),
            "score": d.metadata.get("retrieve_score"),
            "content": d.page_content,
            "doc_kind": d.metadata.get("doc_kind"),
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
    """图节点：模型只做证据标注，类型/二分类由规则层稳定化。"""
    llm = get_deepseek_chat(temperature=0.0)
    payload = {
        "id": state["id"],
        "user_question": state["user_question"],
        "system_reply": state["system_reply"],
        "retrieved_contexts": state.get("retrieved_contexts") or [],
    }
    raw = _invoke_judge_json(llm, payload)
    stable = stabilize_judgment(
        user_question=state["user_question"],
        system_reply=state["system_reply"],
        contexts=list(state.get("retrieved_contexts") or []),
        raw=raw,
    )
    return {
        **state,
        "is_hallucination": bool(stable["is_hallucination"]),
        "hallucination_type": stable.get("hallucination_type"),
        "severity": str(stable.get("severity") or "none"),
        "confidence": float(stable.get("confidence") or 0.0),
        "evidence": str(stable.get("evidence") or ""),
        "detail": str(stable.get("detail") or ""),
        "claims": list(stable.get("claims") or []),
        "mode": "langgraph-chroma-deepseek",
    }


def build_detect_graph():
    """编译「检索 → 判定」LangGraph。

    Returns:
        可 ``invoke`` 的编译图。
    """
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
    gold_hit = any(d.get("doc_id") == gold_id for d in retrieved)
    return {
        "id": item["id"],
        "is_hallucination": bool(out.get("is_hallucination")),
        "hallucination_type": out.get("hallucination_type"),
        "severity": out.get("severity", "none"),
        "confidence": float(out.get("confidence") or 0.0),
        "evidence": out.get("evidence") or "",
        "detail": out.get("detail") or "",
        "claims": out.get("claims") or [],
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
    graph = build_detect_graph()
    return [
        detect_one_rag(item, top_k=top_k, exclude_self=exclude_self, graph=graph)
        for item in items
    ]
