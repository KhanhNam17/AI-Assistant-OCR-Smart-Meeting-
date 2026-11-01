"""Build shareable benchmark reports (Markdown + JSON) summarizing a single run:
inputs used, raw metric values, and the full criteria table with Đạt/Không đạt status.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no extra deps)."""
    columns = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown_report(
    meta: dict[str, Any],
    results: dict[str, Optional[float]],
    rows: list[dict],
    n_pass: int,
    n_fail: int,
    n_na: int,
) -> str:
    """Build a human-readable Markdown report for a benchmark run."""
    df = pd.DataFrame(rows)

    parts: list[str] = []
    parts.append("# Báo cáo Benchmark - Smart Meeting")
    parts.append("")
    parts.append(f"- Thời gian chạy: {meta.get('timestamp')}")
    parts.append(f"- Audio: {meta.get('audio_name', 'N/A')} (thời lượng: {meta.get('audio_duration', 'N/A')})")
    parts.append(f"- Điều kiện audio: {meta.get('audio_condition')}")
    parts.append(f"- Số lượt nói (hệ thống / tham chiếu): {meta.get('n_hyp_segments')} / {meta.get('n_ref_segments')}")
    parts.append(f"- Số từ khoá kiểm tra (1.5): {meta.get('n_keywords')}")
    parts.append(f"- Có dữ liệu tóm tắt (4.5): {'Có' if meta.get('has_summary') else 'Không'}")
    parts.append("")
    parts.append("## Tổng quan")
    parts.append("")
    parts.append(f"- **Đạt**: {n_pass}")
    parts.append(f"- **Không đạt**: {n_fail}")
    parts.append(f"- **N/A / Thủ công**: {n_na}")
    parts.append("")

    if not df.empty:
        for group in df["Nhóm"].unique():
            parts.append(f"## {group}")
            parts.append("")
            group_df = df[df["Nhóm"] == group].drop(columns=["Nhóm"])
            parts.append(dataframe_to_markdown(group_df))
            parts.append("")

    parts.append("## Giá trị đo thô (raw metrics)")
    parts.append("")
    parts.append("```json")
    parts.append(json.dumps(results, ensure_ascii=False, indent=2))
    parts.append("```")
    parts.append("")

    if meta.get("hyp_text") or meta.get("ref_text"):
        parts.append("## Transcript đã dùng")
        parts.append("")
        parts.append("### Hệ thống (Hypothesis)")
        parts.append("")
        parts.append("```")
        parts.append(meta.get("hyp_text", ""))
        parts.append("```")
        parts.append("")
        parts.append("### Tham chiếu (Reference)")
        parts.append("")
        parts.append("```")
        parts.append(meta.get("ref_text", ""))
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def build_json_report(
    meta: dict[str, Any],
    results: dict[str, Optional[float]],
    rows: list[dict],
    n_pass: int,
    n_fail: int,
    n_na: int,
) -> str:
    """Build a machine-readable JSON report (for programmatic review)."""
    payload = {
        "meta": meta,
        "summary": {"dat": n_pass, "khong_dat": n_fail, "na_thu_cong": n_na},
        "raw_metrics": results,
        "criteria": rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
