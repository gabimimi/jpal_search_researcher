"""
Fetch a Salesforce report as JSON and convert tabular rows to a pandas DataFrame.

Reports must be in Tabular format (no row groupings). If your report is grouped,
create a Tabular clone in Salesforce or use a different API.

API reference:
  https://developer.salesforce.com/docs/atlas.en-us.api_analytics.meta/api_analytics/
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def fetch_report_json(
    instance_url: str,
    access_token: str,
    report_id: str,
    api_version: str = "59.0",
) -> dict:
    """GET report including detail rows."""
    url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/analytics/reports/{report_id}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"includeDetails": "true"},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Salesforce report {report_id} failed ({r.status_code}): {r.text[:500]}")
    return r.json()


def _cell_value(cell: Any) -> Any:
    if cell is None:
        return ""
    if isinstance(cell, dict):
        if "value" in cell and cell["value"] is not None:
            return cell["value"]
        if "label" in cell and cell["label"] is not None:
            return cell["label"]
    return cell


def _detail_column_labels(report: dict) -> List[str]:
    """Human-readable column labels in detail column order."""
    meta = report.get("reportMetadata") or {}
    raw_cols = meta.get("detailColumns") or []
    labels: List[str] = []
    ext = report.get("reportExtendedMetadata") or {}
    info = ext.get("detailColumnInfo") or {}

    for c in raw_cols:
        if isinstance(c, str):
            col_info = info.get(c) or {}
            labels.append(str(col_info.get("label") or col_info.get("entityColumnName") or c))
        elif isinstance(c, dict):
            labels.append(str(c.get("label") or c.get("name") or c))
        else:
            labels.append(str(c))

    return labels


def _tabular_rows(fact_map: dict) -> List[dict]:
    """Find the first fact-map bucket that looks like tabular detail rows."""
    if not fact_map:
        return []
    for _key, block in fact_map.items():
        if not isinstance(block, dict):
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        first = rows[0]
        if isinstance(first, dict) and "dataCells" in first:
            return rows
    return []


def report_json_to_dataframe(report: dict) -> pd.DataFrame:
    """
    Convert synchronous Analytics report JSON to a DataFrame.
    """
    labels = _detail_column_labels(report)
    fact_map = report.get("factMap") or {}
    rows = _tabular_rows(fact_map)

    if not labels:
        raise ValueError(
            "Report has no detail columns. Use a Tabular report with detail columns visible."
        )
    if not rows:
        raise ValueError(
            "Report returned no detail rows. Use a Tabular report (not summary/matrix only), "
            "or check filters / row limits in Salesforce."
        )

    records: List[Dict[str, Any]] = []
    for row in rows:
        cells = row.get("dataCells") or []
        rec: Dict[str, Any] = {}
        for i, lab in enumerate(labels):
            if i < len(cells):
                rec[lab] = _cell_value(cells[i])
            else:
                rec[lab] = ""
        records.append(rec)

    return pd.DataFrame.from_records(records)
