"""从评估报告生成检出率图表，并同步到 README。"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from kb.evaluation.ground_truth import analysis_notes, build_readme_analysis_section
from kb.evaluation.llm_analysis import generate_misjudgment_analysis
from kb.paths import OUTPUT_DIR, ROOT

DEFAULT_REPORT = OUTPUT_DIR / "evaluation_report_rag.json"
DEFAULT_CHART = OUTPUT_DIR / "metrics_overview.png"
DEFAULT_README = ROOT / "README.md"
DEFAULT_LLM_ANALYSIS = OUTPUT_DIR / "misjudgment_llm.json"

# 按二级标题定位可自动覆盖的章节（不写 HTML 注释标记）
METRICS_HEADING = "## 3. 检出率验证"
ANALYSIS_HEADING = "## 4. 误判原因分析"

_CN_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Arial Unicode MS",
)


def _configure_font() -> str:
    """选择可用中文字体；找不到则回退 DejaVu（图内用英文标签）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return ""


def render_metrics_chart(
    report: dict[str, Any],
    out_path: Path = DEFAULT_CHART,
) -> Path:
    """绘制混淆矩阵 + 关键指标条形图。

    Args:
        report: ``evaluate`` 返回的报告。
        out_path: PNG 输出路径。

    Returns:
        写出的图片路径。
    """
    use_cn = bool(_configure_font())
    m = report["metrics"]
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    detection = float(m.get("detection_rate") or 0)
    precision = float(m.get("precision") or 0)
    binary_acc = float(m.get("binary_accuracy") or 0)
    exact_acc = float(m.get("exact_accuracy") or m.get("accuracy") or 0)
    type_acc = float(m.get("type_accuracy") or 0)

    if use_cn:
        title = "检出率总览（对照 ground_truth）"
        cm_title = "混淆矩阵（是否幻觉）"
        bar_title = "关键指标"
        labels_cm = [["TP\n已发现", "FP\n误报"], ["FN\n漏检", "TN\n正确放过"]]
        bar_names = ["检出率", "精确率", "二分类准确率", "严格准确率", "类型命中率"]
        xlabel = "比例"
    else:
        title = "Detection metrics vs ground_truth"
        cm_title = "Confusion matrix (binary)"
        bar_title = "Key metrics"
        labels_cm = [["TP", "FP"], ["FN", "TN"]]
        bar_names = ["Recall", "Precision", "Binary Acc", "Exact Acc", "Type Acc"]
        xlabel = "Ratio"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 左：2x2 混淆矩阵热力
    ax0 = axes[0]
    matrix = [[tp, fp], [fn, tn]]
    colors = [["#e6f5ec", "#fff6dd"], ["#fdecea", "#e8f1f8"]]
    for i in range(2):
        for j in range(2):
            ax0.add_patch(
                Rectangle(
                    (j, 1 - i),
                    1,
                    1,
                    facecolor=colors[i][j],
                    edgecolor="#cfc7bb",
                    linewidth=1.2,
                )
            )
            ax0.text(
                j + 0.5,
                1 - i + 0.62,
                labels_cm[i][j],
                ha="center",
                va="center",
                fontsize=10,
                color="#6b645a",
            )
            ax0.text(
                j + 0.5,
                1 - i + 0.28,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color="#1c1915",
            )
    ax0.set_xlim(0, 2)
    ax0.set_ylim(0, 2)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_aspect("equal")
    ax0.set_title(cm_title, fontsize=11)

    # 右：指标条形图
    ax1 = axes[1]
    values = [detection, precision, binary_acc, exact_acc, type_acc]
    bar_colors = ["#2a9d8f", "#245b8a", "#1f7a4c", "#9a6700", "#0f4c5c"]
    y_pos = list(range(len(bar_names)))[::-1]
    bars = ax1.barh(y_pos, values, color=bar_colors, height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(bar_names)
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel(xlabel)
    ax1.set_title(bar_title, fontsize=11)
    ax1.axvline(1.0, color="#e4ddd2", linewidth=1, linestyle="--")
    for bar, val in zip(bars, values, strict=True):
        ax1.text(
            min(val + 0.02, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return out_path


def _fmt_ids(items: list[dict[str, Any]], key: str = "id") -> str:
    if not items:
        return "无"
    return "、".join(str(x.get(key)) for x in items)


def build_metrics_section(
    report: dict[str, Any],
    *,
    chart_rel: str = "output/metrics_overview.png",
    gold_hits: str | None = None,
    html_rel: str = "output/report.html",
    json_rel: str = "output/evaluation_report_rag.json",
) -> str:
    """根据评估报告生成 README「检出率」Markdown 区块正文。

    Args:
        report: ``evaluate`` 返回的报告。
        chart_rel: 图表相对 README 的路径。
        gold_hits: 对应知识 Top-K 命中描述，如 ``18/20``；可选。
        html_rel: HTML 报告相对路径。
        json_rel: JSON 报告相对路径。

    Returns:
        不含同步标记的 Markdown 文本。
    """
    m = report["metrics"]
    fn = report.get("false_negatives") or []
    fp = report.get("false_positives") or []
    mismatch = report.get("type_mismatch") or []
    mismatch_ids = _fmt_ids(mismatch)

    lines = [
        "## 3. 检出率验证（vs ground_truth）",
        "",
        "以下指标由最近一次 `main.py` 对照 `ground_truth` 自动生成。",
        "",
        f"![检出率总览]({chart_rel})",
        "",
        "|  | 预测：幻觉 | 预测：正常 |",
        "| --- | ---: | ---: |",
        f"| **实际：幻觉** | TP **{m['tp']}** | FN **{m['fn']}** |",
        f"| **实际：正常** | FP **{m['fp']}** | TN **{m['tn']}** |",
        "",
        "| 指标 | 数值 | 说明 |",
        "| ---- | ---- | ---- |",
        f"| 检出率 (Recall) | **{float(m['detection_rate']):.1%}** | 有问题的是否都被发现 |",
        f"| 精确率 | **{float(m['precision']):.1%}** | 报出来的是否冤枉好人 |",
        f"| 二分类准确率 | **{float(m.get('binary_accuracy') or 0):.1%}** | 只看「是否幻觉」 |",
        f"| 严格准确率 | **{float(m.get('exact_accuracy') or m.get('accuracy') or 0):.1%}** "
        "| 是否幻觉 + 类型都对 |",
        f"| 类型命中率 | **{float(m.get('type_accuracy') or 0):.1%}** | 已检出样本中类型一致比例 |",
        "",
        f"- **漏检 (FN={m['fn']})**：{_fmt_ids(fn)}",
        f"- **误报 (FP={m['fp']})**：{_fmt_ids(fp)}",
        f"- **类型识别错误（{len(mismatch)}）**：{mismatch_ids}"
        + ("（已检出有问题，但类型判错）" if mismatch else ""),
    ]
    if gold_hits:
        lines.append(f"- 对应知识出现在 Top-K：{gold_hits}")
    lines.extend(
        [
            "",
            f"详细 JSON：[`{json_rel}`]({json_rel})；"
            f"HTML：[`{html_rel}`]({html_rel})。",
            "",
        ]
    )
    return "\n".join(lines)


def sync_readme_section(
    heading_prefix: str,
    section_body: str,
    readme_path: Path = DEFAULT_README,
) -> Path:
    """按二级标题替换 README 中对应整节（直到下一个 ``## ``）。

    Args:
        heading_prefix: 节标题前缀，如 ``## 3. 检出率验证``。
        section_body: 完整节正文（须以该标题开头）。
        readme_path: README 路径。

    Returns:
        更新后的 README 路径。

    Raises:
        ValueError: 找不到对应标题。
    """
    text = readme_path.read_text(encoding="utf-8")
    # 清掉历史 HTML 同步注释，避免残留在文件里
    text = re.sub(r"\n?<!-- (?:METRICS|ANALYSIS):(BEGIN|END) -->\n?", "\n", text)

    pattern = re.compile(rf"(?ms)^({re.escape(heading_prefix)}[^\n]*\n).*?(?=^## |\Z)")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"{readme_path} 缺少以「{heading_prefix}」开头的章节")

    body = section_body.rstrip() + "\n\n"
    text = pattern.sub(body, text, count=1)
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def sync_readme_metrics(
    section_body: str,
    readme_path: Path = DEFAULT_README,
) -> Path:
    """将检出率整节写入 README。"""
    return sync_readme_section(METRICS_HEADING, section_body, readme_path)


def sync_readme_analysis(
    section_body: str,
    readme_path: Path = DEFAULT_README,
) -> Path:
    """将误判分析/建议整节写入 README。"""
    return sync_readme_section(ANALYSIS_HEADING, section_body, readme_path)


def load_or_generate_llm_analysis(
    report: dict[str, Any],
    *,
    path: Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """读取已有 LLM 分析，或重新调用模型生成。

    Args:
        report: 评估报告。
        path: ``misjudgment_llm.json`` 路径。
        refresh: 为 True 时强制重新调用 LLM。

    Returns:
        结构化误判分析结果。
    """
    path = path or DEFAULT_LLM_ANALYSIS
    if not refresh and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    analysis = generate_misjudgment_analysis(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return analysis


def update_metrics_artifacts(
    report: dict[str, Any],
    *,
    out_dir: Path | None = None,
    gold_hits: str | None = None,
    sync_readme: bool = True,
    llm_analysis: dict[str, Any] | None = None,
    refresh_analysis: bool = False,
) -> dict[str, Path]:
    """生成图表并（可选）同步 README 检出率与误判分析区块。

    Args:
        report: 评估报告。
        out_dir: 输出目录，默认 ``output/``。
        gold_hits: 对应知识 Top-K 命中描述。
        sync_readme: 是否写回 README。
        llm_analysis: 已生成的 LLM 分析；缺省则读/生成 ``misjudgment_llm.json``。
        refresh_analysis: 无现成分析对象时是否强制重调 LLM。

    Returns:
        生成产物路径字典（chart / readme / llm_analysis）。
    """
    out_dir = out_dir or OUTPUT_DIR
    chart_path = out_dir / "metrics_overview.png"
    # 先画检出率总览图，再按标题覆盖 README §3/§4
    render_metrics_chart(report, chart_path)
    paths: dict[str, Path] = {"chart": chart_path}

    if sync_readme:
        llm_path = out_dir / "misjudgment_llm.json"
        if llm_analysis is None:
            llm_analysis = load_or_generate_llm_analysis(
                report,
                path=llm_path,
                refresh=refresh_analysis,
            )
        else:
            llm_path.write_text(
                json.dumps(llm_analysis, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        paths["llm_analysis"] = llm_path
        analysis_md = out_dir / "misjudgment_analysis.md"
        analysis_md.write_text(
            analysis_notes(report, llm_analysis) + "\n",
            encoding="utf-8",
        )
        paths["analysis_md"] = analysis_md
        sync_readme_metrics(build_metrics_section(report, gold_hits=gold_hits))
        sync_readme_analysis(build_readme_analysis_section(report, llm_analysis))
        paths["readme"] = DEFAULT_README
    return paths


def main() -> None:
    """CLI：从已有 evaluation JSON 刷新图表与 README。"""
    import argparse

    parser = argparse.ArgumentParser(description="同步检出率图表到 README")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--gold-hits", type=str, default=None)
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="强制重新调用 LLM 生成误判分析",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    paths = update_metrics_artifacts(
        report,
        out_dir=args.out_dir,
        gold_hits=args.gold_hits,
        sync_readme=not args.no_readme,
        refresh_analysis=args.refresh_analysis,
    )
    print(f"[ok] 图表: {paths['chart']}")
    if "readme" in paths:
        print(f"[ok] README 已同步: {paths['readme']}")
    if "llm_analysis" in paths:
        print(f"[ok] LLM 分析: {paths['llm_analysis']}")


if __name__ == "__main__":
    main()
