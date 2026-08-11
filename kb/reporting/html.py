#!/usr/bin/env python3
"""生成面向业务的 HTML 检测报告（浏览器打开即可）。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from kb.paths import DATA_DIR, OUTPUT_DIR

RESULT_LABEL = {
    "TP": "有问题 · 已发现",
    "TN": "没问题 · 判断正确",
    "FP": "没问题 · 被误报",
    "FN": "有问题 · 未发现",
}

SEVERITY_LABEL = {
    "critical": "极高",
    "high": "高",
    "medium": "中",
    "low": "低",
    "none": "无",
}


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def build_html(
    detections: list[dict[str, Any]],
    replies: list[dict[str, Any]],
    report: dict[str, Any],
) -> str:
    reply_map = {r["id"]: r for r in replies}
    m = report["metrics"]
    per = {c["id"]: c for c in report.get("per_case", [])}
    type_counts = report.get("by_type", {}).get("ground_truth_counts", {})
    mismatches = report.get("type_mismatch", [])
    false_negatives = report.get("false_negatives") or []
    false_positives = report.get("false_positives") or []

    max_type = max(type_counts.values()) if type_counts else 1
    bars = "".join(
        f"""
        <div class="bar-row">
          <span class="bar-label">{_esc(name)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{v / max_type * 100:.0f}%"></div></div>
          <span class="bar-n">{v}</span>
        </div>"""
        for name, v in sorted(type_counts.items(), key=lambda x: -x[1])
    )

    def _error_rows(
        items: list[dict[str, Any]],
        *,
        type_key: str,
        empty: str,
    ) -> str:
        if not items:
            return f'<tr><td colspan="4" class="muted">{_esc(empty)}</td></tr>'
        return "".join(
            f"""
        <tr>
          <td><code>{_esc(x["id"])}</code></td>
          <td>{_esc(x.get(type_key) or "—")}</td>
          <td class="muted">{_esc((x.get("gt_detail") or "")[:100])}</td>
          <td class="muted">{_esc((x.get("pred_detail") or "")[:100])}</td>
        </tr>"""
            for x in items
        )

    fn_rows = _error_rows(
        false_negatives, type_key="gt_type", empty="本轮没有漏检"
    )
    fp_rows = _error_rows(
        false_positives, type_key="pred_type", empty="本轮没有误报"
    )
    # 无漏检/误报时不渲染空表，避免占版面
    if false_negatives or false_positives:
        binary_err_section = f"""
    <h2>二分类错误（漏检 / 误报）</h2>
    <h3>漏检 (FN)</h3>
    <table>
      <thead>
        <tr>
          <th>编号</th><th>标注类型</th><th>标注说明</th><th>模型说明</th>
        </tr>
      </thead>
      <tbody>{fn_rows}</tbody>
    </table>
    <h3>误报 (FP)</h3>
    <table>
      <thead>
        <tr>
          <th>编号</th><th>模型类型</th><th>标注说明</th><th>模型说明</th>
        </tr>
      </thead>
      <tbody>{fp_rows}</tbody>
    </table>
"""
    else:
        binary_err_section = ""
    mm_rows = (
        "".join(
            f"""
        <tr>
          <td><code>{_esc(x["id"])}</code></td>
          <td>{_esc(x.get("gt_type"))}</td>
          <td>{_esc(x.get("pred_type"))}</td>
          <td class="muted">{_esc((x.get("gt_detail") or "")[:100])}</td>
        </tr>"""
            for x in mismatches
        )
        or '<tr><td colspan="4" class="muted">本轮没有类型识别错误</td></tr>'
    )

    case_rows = []
    for d in detections:
        rid = d["id"]
        item = reply_map.get(rid, {})
        c = per.get(rid, {})
        label = c.get("label", "?")
        badge = {"TP": "ok", "TN": "info", "FP": "warn", "FN": "bad"}.get(label, "")
        result_text = RESULT_LABEL.get(label, label)
        human_t = c.get("gt_type") or "正常"
        system_t = c.get("pred_type") or "正常"
        if human_t == system_t:
            type_cell = human_t if c.get("gt_hallucination") else "正常"
            type_flag = "same"
        else:
            type_cell = f"人工：{human_t}｜系统：{system_t}"
            type_flag = "diff"
        gold = (d.get("retrieval") or {}).get("gold_in_retrieved")
        gold_s = "已找到" if gold else ("未找到" if gold is False else "—")
        sev = SEVERITY_LABEL.get(
            str(d.get("severity") or ""), str(d.get("severity") or "—")
        )
        case_rows.append(
            f"""
        <tr data-result="{badge}" data-typediff="{type_flag}">
          <td><code>{_esc(rid)}</code></td>
          <td><span class="badge {badge}">{_esc(result_text)}</span></td>
          <td>{_esc(item.get("user_question"))}</td>
          <td>{_esc(type_cell)}</td>
          <td>{_esc(sev)}</td>
          <td>{_esc(gold_s)}</td>
          <td class="muted">{_esc((d.get("detail") or "")[:110])}</td>
        </tr>"""
        )

    detection_rate = float(m.get("detection_rate") or 0)
    binary_accuracy = float(m.get("binary_accuracy") or m.get("accuracy") or 0)
    exact_accuracy = float(m.get("exact_accuracy") or m.get("accuracy") or 0)
    type_accuracy = float(m.get("type_accuracy") or 0)
    type_mismatch_n = int(m.get("type_mismatch_count") or len(mismatches))
    fp_n = int(m.get("fp") or len(false_positives))
    fn_n = int(m.get("fn") or len(false_negatives))
    total_n = int(m.get("total") or len(detections))
    has_binary_err = fp_n > 0 or fn_n > 0
    banner_cls = "banner warn" if (has_binary_err or type_mismatch_n) else "banner"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>客服回复质量检测报告</title>
<style>
  :root {{
    --bg: #f6f4ef;
    --card: #fffdf8;
    --ink: #1c1915;
    --muted: #6b645a;
    --line: #e4ddd2;
    --ok: #1f7a4c;
    --ok-bg: #e6f5ec;
    --info: #245b8a;
    --info-bg: #e8f1f8;
    --warn: #9a6700;
    --warn-bg: #fff6dd;
    --bad: #b42318;
    --bad-bg: #fdecea;
    --accent: #0f4c5c;
    --fill: #2a9d8f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", system-ui, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.5;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  h2 {{ font-size: 18px; margin: 28px 0 12px; }}
  .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 24px; }}
  .banner {{
    background: var(--ok-bg); border: 1px solid #b7e0c6; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 20px; color: var(--ok); font-weight: 600;
  }}
  .banner.warn {{
    background: var(--warn-bg); border-color: #edd48a; color: var(--warn);
  }}
  .stats {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;
  }}
  .stat {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 16px;
  }}
  .stat .v {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .stat .l {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 16px;
  }}
  .matrix {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; text-align: center;
  }}
  .matrix div {{
    border-radius: 8px; padding: 14px 8px; background: #f0ebe3;
  }}
  .matrix .ok {{ background: var(--ok-bg); color: var(--ok); }}
  .matrix .info {{ background: var(--info-bg); color: var(--info); }}
  .matrix .n {{ font-size: 26px; font-weight: 700; }}
  .matrix .t {{ font-size: 12px; opacity: .85; }}
  .bar-row {{ display: grid; grid-template-columns: 88px 1fr 24px; gap: 8px; align-items: center; margin: 6px 0; }}
  .bar-label {{ font-size: 12px; color: var(--muted); }}
  .bar-track {{ height: 10px; background: #efeae2; border-radius: 99px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--fill); border-radius: 99px; }}
  .bar-n {{ font-size: 12px; font-weight: 600; text-align: right; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 10px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  th {{ background: #f0ebe3; font-size: 12px; color: var(--muted); font-weight: 600; position: sticky; top: 0; }}
  tr:last-child td {{ border-bottom: 0; }}
  .muted {{ color: var(--muted); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 700;
  }}
  .badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .badge.info {{ background: var(--info-bg); color: var(--info); }}
  .badge.warn {{ background: var(--warn-bg); color: var(--warn); }}
  .badge.bad {{ background: var(--bad-bg); color: var(--bad); }}
  code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; }}
  .filters {{ display: flex; gap: 8px; margin: 8px 0 12px; flex-wrap: wrap; }}
  .filters button {{
    border: 1px solid var(--line); background: var(--card); border-radius: 99px;
    padding: 6px 12px; cursor: pointer; font-size: 12px; color: var(--ink);
  }}
  .filters button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  @media (max-width: 800px) {{
    .stats, .grid2 {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media (max-width: 520px) {{
    .stats, .grid2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>客服回复质量检测报告</h1>
    <div class="sub">共检测 {m.get("total", len(detections))} 条客服回复 · 对照人工复核结果</div>

    <div class="{banner_cls}">
      {total_n} 条：二分类准确率 {binary_accuracy:.0%}（检出率 {detection_rate:.0%}）；
      漏检 {fn_n} 条、误报 {fp_n} 条；
      类型识别错误 {type_mismatch_n} 条；严格准确率 {exact_accuracy:.0%}，类型命中率 {type_accuracy:.0%}。
    </div>

    <div class="stats">
      <div class="stat"><div class="v">{m.get("tp", 0)}</div><div class="l">有问题 · 已发现 (TP)</div></div>
      <div class="stat"><div class="v">{m.get("tn", 0)}</div><div class="l">没问题 · 判断正确 (TN)</div></div>
      <div class="stat"><div class="v">{fn_n}</div><div class="l">漏检 (FN)</div></div>
      <div class="stat"><div class="v">{fp_n}</div><div class="l">误报 (FP)</div></div>
    </div>
    <p class="sub" style="margin-top:-8px;margin-bottom:20px">
      二分类错误 = 漏检 + 误报；类型识别错误另计（是否幻觉对了但类型判错），计入严格准确率扣分。
      本轮类型识别错误 {type_mismatch_n} 条，严格准确率 {exact_accuracy:.0%}。
    </p>

    <div class="grid2">
      <div class="card">
        <h2 style="margin-top:0">四种结果分别多少条</h2>
        <div class="matrix">
          <div class="ok"><div class="n">{m.get("tp", 0)}</div><div class="t">有问题 · 已发现</div></div>
          <div><div class="n">{m.get("fp", 0)}</div><div class="t">没问题 · 被误报</div></div>
          <div><div class="n">{m.get("fn", 0)}</div><div class="t">有问题 · 未发现</div></div>
          <div class="info"><div class="n">{m.get("tn", 0)}</div><div class="t">没问题 · 判断正确</div></div>
        </div>
      </div>
      <div class="card">
        <h2 style="margin-top:0">问题类型分布（人工复核）</h2>
        {bars}
      </div>
    </div>

    {binary_err_section}
    <h2>类型识别错误的条目（是否幻觉已判对，类型判错）</h2>
    <table>
      <thead>
        <tr>
          <th>编号</th><th>人工分类</th><th>系统分类</th><th>说明</th>
        </tr>
      </thead>
      <tbody>{mm_rows}</tbody>
    </table>

    <h2>逐条明细</h2>
    <div class="filters" id="filters">
      <button class="active" data-f="all">全部</button>
      <button data-f="ok">有问题 · 已发现</button>
      <button data-f="info">没问题 · 判断正确</button>
      <button data-f="bad">漏检</button>
      <button data-f="warn">误报</button>
      <button data-f="typediff">类型识别错误</button>
    </div>
    <table id="cases">
      <thead>
        <tr>
          <th>编号</th><th>检测结论</th><th>用户问题</th><th>问题类型</th>
          <th>严重程度</th><th>相关知识</th><th>说明</th>
        </tr>
      </thead>
      <tbody>
        {"".join(case_rows)}
      </tbody>
    </table>
  </div>
<script>
  const btns = document.querySelectorAll('#filters button');
  const rows = [...document.querySelectorAll('#cases tbody tr')];
  btns.forEach(b => b.addEventListener('click', () => {{
    btns.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = b.dataset.f;
    rows.forEach(r => {{
      const result = r.dataset.result || '';
      const typediff = r.dataset.typediff || '';
      let show = true;
      if (f === 'ok') show = result === 'ok';
      if (f === 'info') show = result === 'info';
      if (f === 'bad') show = result === 'bad';
      if (f === 'warn') show = result === 'warn';
      if (f === 'typediff') show = typediff === 'diff';
      r.style.display = show ? '' : 'none';
    }});
  }}));
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 HTML 幻觉检测报告")
    parser.add_argument(
        "--detections",
        type=Path,
        default=OUTPUT_DIR / "detection_results_rag.json",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=OUTPUT_DIR / "evaluation_report_rag.json",
    )
    parser.add_argument("--replies", type=Path, default=DATA_DIR / "replies.json")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "report.html")
    args = parser.parse_args()

    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    replies = json.loads(args.replies.read_text(encoding="utf-8"))
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(detections, replies, report), encoding="utf-8")
    print(f"[ok] HTML 报告已生成: {args.out}")
    print(f"     浏览器打开: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
