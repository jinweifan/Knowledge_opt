"""用 LLM 根据本轮评测错误生成误判分析（非硬编码剧本）。"""

from __future__ import annotations

import json
import re
from typing import Any

from kb.pipeline.llm import get_deepseek_chat
from kb.prompts.loader import render_prompt


def _parse_json(content: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象。"""
    content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError(f"无法从模型输出解析 JSON：{content[:200]}") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("误判分析输出必须是 JSON 对象")
    return data


def _errors_payload(report: dict[str, Any]) -> dict[str, Any]:
    """从评估报告抽出供分析用的错误事实。"""
    return {
        "metrics_summary": {
            "fn": len(report.get("false_negatives") or []),
            "fp": len(report.get("false_positives") or []),
            "type_mismatch": len(report.get("type_mismatch") or []),
            "binary_accuracy": (report.get("metrics") or {}).get("binary_accuracy"),
            "exact_accuracy": (report.get("metrics") or {}).get("exact_accuracy"),
            "type_accuracy": (report.get("metrics") or {}).get("type_accuracy"),
        },
        "false_negatives": report.get("false_negatives") or [],
        "false_positives": report.get("false_positives") or [],
        "type_mismatch": report.get("type_mismatch") or [],
    }


def _index_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(x.get("id")): x for x in items if x.get("id") is not None}


def _normalize_analysis(
    raw: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """校验并规范化 LLM 输出，缺项用空说明占位。"""
    expected_mm = {str(x["id"]) for x in report.get("type_mismatch") or []}
    expected_fn = {str(x["id"]) for x in report.get("false_negatives") or []}
    expected_fp = {str(x["id"]) for x in report.get("false_positives") or []}

    mm_map = _index_by_id(list(raw.get("type_mismatch") or []))
    fn_map = _index_by_id(list(raw.get("false_negatives") or []))
    fp_map = _index_by_id(list(raw.get("false_positives") or []))

    type_mismatch = []
    for rid in sorted(expected_mm, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        item = mm_map.get(rid) or {}
        type_mismatch.append(
            {
                "id": rid,
                "reason": str(item.get("reason") or "").strip() or "（模型未给出判错原因）",
                "reason_short": str(item.get("reason_short") or item.get("reason") or "")
                .strip()
                or "（未给出）",
            }
        )

    false_negatives = []
    for rid in sorted(expected_fn, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        item = fn_map.get(rid) or {}
        false_negatives.append(
            {
                "id": rid,
                "reason": str(item.get("reason") or "").strip() or "（模型未给出漏检原因）",
            }
        )

    false_positives = []
    for rid in sorted(expected_fp, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        item = fp_map.get(rid) or {}
        false_positives.append(
            {
                "id": rid,
                "reason": str(item.get("reason") or "").strip() or "（模型未给出误报原因）",
            }
        )

    pitfalls = [str(x).strip() for x in (raw.get("pitfalls") or []) if str(x).strip()]
    takeaways = [str(x).strip() for x in (raw.get("takeaways") or []) if str(x).strip()]
    if not pitfalls:
        pitfalls = ["（本轮模型未给出易误判归纳）"]
    if not takeaways:
        takeaways = ["（本轮模型未给出看法/改进方向）"]

    return {
        "type_mismatch": type_mismatch,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "pitfalls": pitfalls,
        "takeaways": takeaways,
    }


def generate_misjudgment_analysis(report: dict[str, Any]) -> dict[str, Any]:
    """调用 DeepSeek，根据本轮错误清单生成误判分析结构化结果。

    Args:
        report: ``evaluate`` 返回的评估报告。

    Returns:
        规范化后的分析结果（含 type_mismatch / pitfalls / takeaways 等）。

    Raises:
        RuntimeError: 未配置 API Key。
        ValueError: 模型输出无法解析为有效 JSON。
    """
    payload = _errors_payload(report)
    # 无任何错误时仍让模型写简短看法，避免空跑硬编码文案
    prompt = render_prompt(
        "misjudgment_analysis",
        errors_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    llm = get_deepseek_chat(temperature=0.3)
    msg = llm.invoke(
        [
            {
                "role": "system",
                "content": "你只输出合法 JSON，不要编造输入中不存在的样本。",
            },
            {"role": "user", "content": prompt},
        ]
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    raw = _parse_json(content)
    return _normalize_analysis(raw, report)
