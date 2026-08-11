"""报告呈现：HTML、图表、README 同步。"""

from __future__ import annotations

__all__ = ["build_html", "update_metrics_artifacts"]


def __getattr__(name: str):
    """延迟导入，避免 ``python -m kb.reporting.sync`` 与包初始化冲突。"""
    if name == "build_html":
        from kb.reporting.html import build_html

        return build_html
    if name == "update_metrics_artifacts":
        from kb.reporting.sync import update_metrics_artifacts

        return update_metrics_artifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
