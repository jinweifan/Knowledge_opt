"""对照 ground_truth 评估检出率、漏检与误报。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 列表文件。

    Args:
        path: JSON 文件路径。

    Returns:
        反序列化后的字典列表。
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    """对照标准答案计算二分类与类型相关指标。

    Args:
        predictions: 模型预测（需含 id / is_hallucination 等）。
        ground_truth: 标准答案标注。

    Returns:
        含 metrics、漏检/误报列表、per_case 等字段的报告。
    """
    gt_map = {g["id"]: g for g in ground_truth}
    pred_map = {p["id"]: p for p in predictions}

    ids = sorted(set(gt_map) | set(pred_map), key=lambda x: int(x[1:]))

    # 二分类混淆矩阵计数；类型错误另记（仍属 TP）
    tp = fp = tn = fn = 0
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    type_mismatch: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    for rid in ids:
        gt = gt_map.get(rid)
        pred = pred_map.get(rid)
        if not gt or not pred:
            continue
        gt_h = bool(gt["is_hallucination"])
        pred_h = bool(pred["is_hallucination"])

        if gt_h and pred_h:
            # 有问题且已发现
            tp += 1
            label = "TP"
            # 是否幻觉对了，但细类不同 → 类型识别错误
            if (gt.get("hallucination_type") or None) != (
                pred.get("hallucination_type") or None
            ):
                type_mismatch.append(
                    {
                        "id": rid,
                        "gt_type": gt.get("hallucination_type"),
                        "pred_type": pred.get("hallucination_type"),
                        "gt_detail": gt.get("detail"),
                        "pred_detail": pred.get("detail"),
                    }
                )
        elif (not gt_h) and (not pred_h):
            # 没问题且判断正确
            tn += 1
            label = "TN"
        elif (not gt_h) and pred_h:
            # 误报：标准答案正常，模型却判幻觉
            fp += 1
            label = "FP"
            false_positives.append(
                {
                    "id": rid,
                    "pred_type": pred.get("hallucination_type"),
                    "pred_detail": pred.get("detail"),
                    "gt_detail": gt.get("detail"),
                }
            )
        else:
            # 漏检：标准答案有幻觉，模型未发现
            fn += 1
            label = "FN"
            false_negatives.append(
                {
                    "id": rid,
                    "gt_type": gt.get("hallucination_type"),
                    "gt_detail": gt.get("detail"),
                    "pred_detail": pred.get("detail"),
                }
            )

        per_case.append(
            {
                "id": rid,
                "label": label,
                "gt_hallucination": gt_h,
                "pred_hallucination": pred_h,
                "gt_type": gt.get("hallucination_type"),
                "pred_type": pred.get("hallucination_type"),
            }
        )

    # 聚合指标：检出率=召回；二分类 vs 严格（含类型）两套口径
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # 检出率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # 二分类：只看「是否幻觉」，类型分错仍算对
    binary_accuracy = (tp + tn) / total if total else 0.0
    # 严格：是否幻觉 + 类型都对；类型错从 TP 中扣除
    type_mismatch_n = len(type_mismatch)
    exact_correct = tn + (tp - type_mismatch_n)
    exact_accuracy = exact_correct / total if total else 0.0
    type_accuracy = (tp - type_mismatch_n) / tp if tp else 1.0  # 已检出样本中类型命中率

    # 按标准答案类型统计检出/漏检，便于看哪类难检
    gt_type_counts = Counter(
        g["hallucination_type"] for g in ground_truth if g["is_hallucination"]
    )
    detected_by_type: Counter[str] = Counter()
    missed_by_type: Counter[str] = Counter()
    for case in per_case:
        gt_type = case["gt_type"]
        if not case["gt_hallucination"]:
            continue
        if case["label"] == "TP":
            detected_by_type[gt_type] += 1
        elif case["label"] == "FN":
            missed_by_type[gt_type] += 1

    return {
        "metrics": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "detection_rate": round(recall, 4),
            "f1": round(f1, 4),
            "binary_accuracy": round(binary_accuracy, 4),
            # accuracy 与 exact_accuracy 同为严格口径，兼容旧字段名
            "accuracy": round(exact_accuracy, 4),
            "exact_accuracy": round(exact_accuracy, 4),
            "type_accuracy": round(type_accuracy, 4),
            "type_mismatch_count": type_mismatch_n,
            "exact_correct": exact_correct,
            "total": total,
            "gt_positive": sum(1 for g in ground_truth if g["is_hallucination"]),
            "gt_negative": sum(1 for g in ground_truth if not g["is_hallucination"]),
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "type_mismatch": type_mismatch,
        "by_type": {
            "ground_truth_counts": dict(gt_type_counts),
            "detected_counts": dict(detected_by_type),
            "missed_counts": dict(missed_by_type),
        },
        "per_case": per_case,
    }


def _reason_maps(
    llm_analysis: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """从 LLM 分析结果抽出 id→原因映射。"""
    mm_long = {
        str(x["id"]): str(x.get("reason") or "")
        for x in llm_analysis.get("type_mismatch") or []
    }
    mm_short = {
        str(x["id"]): str(x.get("reason_short") or x.get("reason") or "")
        for x in llm_analysis.get("type_mismatch") or []
    }
    fn_map = {
        str(x["id"]): str(x.get("reason") or "")
        for x in llm_analysis.get("false_negatives") or []
    }
    fp_map = {
        str(x["id"]): str(x.get("reason") or "")
        for x in llm_analysis.get("false_positives") or []
    }
    return mm_long, mm_short, fn_map, fp_map


def analysis_notes(report: dict[str, Any], llm_analysis: dict[str, Any]) -> str:
    """用评估事实 + LLM 分析拼出误判原因 Markdown。

    Args:
        report: ``evaluate`` 返回的报告。
        llm_analysis: ``generate_misjudgment_analysis`` 的结构化结果。

    Returns:
        可读的误判分析说明。
    """
    lines = ["## 误判原因分析", ""]
    fn = report["false_negatives"]
    fp = report["false_positives"]
    mismatch = report["type_mismatch"]
    mm_long, _, fn_map, fp_map = _reason_maps(llm_analysis)

    if not fn and not fp:
        lines.append("本轮二分类（是否幻觉）无漏检、无误报。")
    else:
        if fn:
            lines.append("### 漏检 (False Negative)")
            for item in fn:
                rid = str(item["id"])
                lines.append(
                    f"- **{rid}**（GT: {item['gt_type']}）：{item['gt_detail']}"
                )
                lines.append(f"  - 模型输出：{item.get('pred_detail')}")
                lines.append(f"  - 漏检原因：{fn_map.get(rid) or '（无）'}")
            lines.append("")
        if fp:
            lines.append("### 误报 (False Positive)")
            for item in fp:
                rid = str(item["id"])
                lines.append(
                    f"- **{rid}**（Pred: {item['pred_type']}）：{item['gt_detail']}"
                )
                lines.append(f"  - 模型输出：{item.get('pred_detail')}")
                lines.append(f"  - 误报原因：{fp_map.get(rid) or '（无）'}")
            lines.append("")

    if mismatch:
        lines.append("### 类型识别错误（已检出有问题，但类型判错，严格口径不算全对）")
        for item in mismatch:
            rid = str(item["id"])
            lines.append(
                f"- **{rid}**：GT=`{item['gt_type']}` vs Pred=`{item['pred_type']}`"
            )
            lines.append(f"  - 标注说明：{item.get('gt_detail')}")
            lines.append(f"  - 模型说明：{item.get('pred_detail')}")
            lines.append(f"  - 判错原因：{mm_long.get(rid) or '（无）'}")
        lines.append("")

    lines.append("### 易误判 case 归纳")
    for i, tip in enumerate(llm_analysis.get("pitfalls") or [], start=1):
        lines.append(f"{i}. {tip}")
    lines.append("")
    lines.append("### 怎么看")
    for i, tip in enumerate(llm_analysis.get("takeaways") or [], start=1):
        lines.append(f"{i}. {tip}")
    return "\n".join(lines)


def build_readme_analysis_section(
    report: dict[str, Any],
    llm_analysis: dict[str, Any],
) -> str:
    """根据评估报告与 LLM 分析生成 README「误判分析」区块。

    Args:
        report: ``evaluate`` 返回的报告。
        llm_analysis: ``generate_misjudgment_analysis`` 的结构化结果。

    Returns:
        Markdown 文本。
    """
    fn = report.get("false_negatives") or []
    fp = report.get("false_positives") or []
    mismatch = report.get("type_mismatch") or []
    _, mm_short, _, _ = _reason_maps(llm_analysis)

    if not fn and not fp:
        summary = (
            f"本轮二分类无漏检/误报；主要问题是**类型识别错误**（{len(mismatch)} 条）。"
            if mismatch
            else "本轮二分类无漏检/误报，类型也全部识别正确。"
        )
    else:
        summary = (
            f"本轮漏检 {len(fn)} 条、误报 {len(fp)} 条；"
            f"类型识别错误 {len(mismatch)} 条。"
        )

    lines = [
        "## 4. 误判原因分析",
        "",
        summary,
        "",
        "> 下方「为何判错」由本轮评测后调用 LLM 生成，非预写文案。",
        "",
        "### 二分类错误预测（是否幻觉判错）",
        "",
    ]

    lines.extend(
        [
            "| 错误类型 | 数量 | 样本 |",
            "| -------- | ---: | ---- |",
            f"| 漏检 (FN) | {len(fn)} | {_fmt_error_ids(fn)} |",
            f"| 误报 (FP) | {len(fp)} | {_fmt_error_ids(fp)} |",
            "",
        ]
    )
    if fn:
        lines.append("**漏检明细**")
        lines.append("")
        lines.append("| Case | 标注类型 | 标注说明 | 模型说明 |")
        lines.append("| ---- | -------- | -------- | -------- |")
        for item in fn:
            lines.append(
                f"| {item['id']} | {item.get('gt_type') or '—'} | "
                f"{_short(item.get('gt_detail'))} | {_short(item.get('pred_detail'))} |"
            )
        lines.append("")
    if fp:
        lines.append("**误报明细**")
        lines.append("")
        lines.append("| Case | 模型类型 | 标注说明 | 模型说明 |")
        lines.append("| ---- | -------- | -------- | -------- |")
        for item in fp:
            lines.append(
                f"| {item['id']} | {item.get('pred_type') or '—'} | "
                f"{_short(item.get('gt_detail'))} | {_short(item.get('pred_detail'))} |"
            )
        lines.append("")

    lines.append("### 类型识别错误（是否幻觉对了，类型判错）")
    lines.append("")
    if mismatch:
        lines.extend(
            [
                "| Case | 标注 | 预测 | 为何判错 |",
                "| ---- | ---- | ---- | -------- |",
            ]
        )
        for item in mismatch:
            rid = str(item["id"])
            why = _short(mm_short.get(rid) or "（无）", n=80)
            lines.append(
                f"| {rid} | {item.get('gt_type')} | {item.get('pred_type')} | {why} |"
            )
        lines.append("")
    else:
        lines.append("本轮无类型识别错误。")
        lines.append("")

    return "\n".join(lines)


def _fmt_error_ids(items: list[dict[str, Any]]) -> str:
    if not items:
        return "无"
    return "、".join(str(x.get("id")) for x in items)


def _short(text: Any, n: int = 60) -> str:
    s = "" if text is None else str(text).replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
