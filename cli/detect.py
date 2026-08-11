#!/usr/bin/env python3
"""客服回复幻觉检测 CLI。

流程：加载回复 → 建索引 → RAG 检测 → 对照标准答案评估 → 落盘报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kb.evaluation import (
    analysis_notes,
    evaluate,
    generate_misjudgment_analysis,
    load_json,
)
from kb.paths import DATA_DIR, INDEX_DIR, OUTPUT_DIR
from kb.pipeline import build_chroma_index, detect_batch_rag
from kb.reporting.html import build_html
from kb.reporting.sync import update_metrics_artifacts
from kb.taxonomy import taxonomy_markdown


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="客服回复幻觉批量检测（RAG）")
    parser.add_argument("--replies", type=Path, default=DATA_DIR / "replies.json")
    parser.add_argument(
        "--ground-truth", type=Path, default=DATA_DIR / "ground_truth.json"
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--exclude-self",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="检索时排除本条对应知识（消融实验）",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="检测前重建 Chroma 索引",
    )
    parser.add_argument("--show-taxonomy", action="store_true")
    return parser.parse_args()


def dump_json(path: Path, data: Any) -> None:
    """写入 UTF-8 JSON（缩进、保留中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_index(replies_path: Path, *, rebuild: bool) -> None:
    """必要时构建或重建向量索引。"""
    # 首次运行或显式要求重建时才建库，避免每次检测都重嵌向量
    if rebuild or not INDEX_DIR.exists():
        print("[info] 构建 Chroma 索引...")
        build_chroma_index(replies_path, reset=True)


def save_detections(out_dir: Path, predictions: list[dict[str, Any]]) -> Path:
    """落盘检测结果（带后缀 + 最新副本），返回主路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "detection_results_rag.json"
    dump_json(det_path, predictions)
    # 无后缀副本方便下游默认读取「最新一次」结果
    dump_json(out_dir / "detection_results.json", predictions)
    print(f"[info] 检测结果已写入 {det_path}")
    return det_path


def print_detections(predictions: list[dict[str, Any]]) -> None:
    """打印逐条检测摘要。"""
    print("\n=== 逐条检测 ===")
    for r in predictions:
        flag = "幻觉" if r.get("is_hallucination") else "正常"
        typ = r.get("hallucination_type") or "-"
        print(
            f"{r['id']}: {flag:4s} | {str(typ):8s} | "
            f"sev={str(r.get('severity')):8s} | "
            f"conf={float(r.get('confidence') or 0):.2f} | mode={r.get('mode')}"
        )
        print(f"{r.get('detail')}")
        retrieval = r.get("retrieval")
        if not retrieval:
            continue
        docs = retrieval.get("docs") or []
        ids = ",".join(str(d.get("doc_id")) for d in docs)
        print(f"      retrieved=[{ids}] exclude_self={retrieval.get('exclude_self')}")


def count_gold_in_topk(predictions: list[dict[str, Any]]) -> int:
    """统计「本条对应知识」出现在检索 Top-K 的次数。"""
    return sum(
        1 for p in predictions if (p.get("retrieval") or {}).get("gold_in_retrieved")
    )


def print_eval_metrics(
    report: dict[str, Any],
    *,
    gold_hits: int,
    total: int,
    exclude_self: bool,
) -> None:
    """打印二分类与类型相关指标。"""
    m = report["metrics"]
    print("\n=== 检出率评估 (vs ground_truth) ===")
    print(
        f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']} | "
        f"检出率(Recall)={m['detection_rate']:.2%} | "
        f"精确率={m['precision']:.2%} | F1={m['f1']:.2%} | "
        f"二分类准确率={m['binary_accuracy']:.2%} | "
        f"严格准确率(含类型)={m['exact_accuracy']:.2%} | "
        f"类型命中率={m['type_accuracy']:.2%}"
    )

    fn_ids = ", ".join(x["id"] for x in report["false_negatives"]) or "无"
    fp_ids = ", ".join(x["id"] for x in report["false_positives"]) or "无"
    print(f"漏检(FN): {fn_ids}")
    print(f"误报(FP): {fp_ids}")

    if report["type_mismatch"]:
        mm = ", ".join(
            f"{x['id']}({x['gt_type']}→{x['pred_type']})"
            for x in report["type_mismatch"]
        )
        print(f"类型识别错误: {mm}")
    else:
        print("类型识别错误: 无")

    print(f"对应知识出现在 Top-K: {gold_hits}/{total} (exclude_self={exclude_self})")


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truth_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """对照标准答案评估，并写入 evaluation_report*.json。"""
    gt = load_json(ground_truth_path)
    # 评测只需要判定字段，去掉 retrieval 等大字段便于落盘对照
    slim = [
        {
            "id": p["id"],
            "is_hallucination": p["is_hallucination"],
            "hallucination_type": p.get("hallucination_type"),
            "detail": p.get("detail"),
        }
        for p in predictions
    ]
    report = evaluate(slim, gt)
    dump_json(out_dir / "evaluation_report_rag.json", report)
    dump_json(out_dir / "evaluation_report.json", report)
    return report


def write_side_reports(
    *,
    predictions: list[dict[str, Any]],
    replies: list[dict[str, Any]],
    report: dict[str, Any],
    out_dir: Path,
    gold_hits: str,
) -> str:
    """写误判分析、HTML、图表，并同步 README；返回分析正文。"""
    # 1) 评测后调用 LLM 写误判分析（非预写剧本）
    print("[info] 调用 LLM 生成误判分析…")
    llm_analysis = generate_misjudgment_analysis(report)
    dump_json(out_dir / "misjudgment_llm.json", llm_analysis)
    analysis = analysis_notes(report, llm_analysis)
    analysis_path = out_dir / "misjudgment_analysis.md"
    analysis_path.write_text(analysis + "\n", encoding="utf-8")
    print(f"[info] 评估报告: {out_dir / 'evaluation_report_rag.json'}")
    print(f"[info] LLM 分析 JSON: {out_dir / 'misjudgment_llm.json'}")
    print(f"[info] 误判分析: {analysis_path}")

    # 2) 可交互 HTML 明细（失败不影响主流程）
    try:
        html_path = out_dir / "report.html"
        html_path.write_text(
            build_html(predictions, replies, report),
            encoding="utf-8",
        )
        print(f"[info] HTML 报告: {html_path}")
        print(f"       浏览器打开: file://{html_path.resolve()}")
    except Exception as e:
        print(f"[warn] HTML 报告生成失败: {e}")

    # 3) 检出率图 + README §3/§4 自动覆盖
    try:
        artifacts = update_metrics_artifacts(
            report,
            out_dir=out_dir,
            gold_hits=gold_hits,
            sync_readme=True,
            llm_analysis=llm_analysis,
        )
        print(f"[info] 检出率图表: {artifacts['chart']}")
        if "readme" in artifacts:
            print(
                f"[info] README 已同步（检出率 + LLM 误判分析）: {artifacts['readme']}"
            )
    except Exception as e:
        print(f"[warn] 检出率图表/README 同步失败: {e}")

    return analysis


def main() -> None:
    """CLI 入口：检测 → 评估 → 报告。"""
    args = parse_args()
    if args.show_taxonomy:
        print(taxonomy_markdown())
        return

    # 加载待测回复
    replies = load_json(args.replies)
    print(f"[info] 加载 {len(replies)} 条回复，pipeline=rag")

    # 向量索引就绪后做 RAG 批量检测（检索 + LLM 判定）
    ensure_index(args.replies, rebuild=args.rebuild_index)
    predictions = detect_batch_rag(
        replies,
        top_k=args.top_k,
        exclude_self=args.exclude_self,
    )
    save_detections(args.out_dir, predictions)
    print_detections(predictions)

    # 对照标准答案算漏检/误报/类型错误，并生成报告
    if not args.ground_truth.exists():
        print(f"[warn] 未找到标准答案: {args.ground_truth}")
        return

    report = evaluate_predictions(predictions, args.ground_truth, args.out_dir)
    gold_hits = count_gold_in_topk(predictions)
    print_eval_metrics(
        report,
        gold_hits=gold_hits,
        total=len(predictions),
        exclude_self=args.exclude_self,
    )
    analysis = write_side_reports(
        predictions=predictions,
        replies=replies,
        report=report,
        out_dir=args.out_dir,
        gold_hits=f"{gold_hits}/{len(predictions)}",
    )
    print("\n" + analysis)


if __name__ == "__main__":
    main()
