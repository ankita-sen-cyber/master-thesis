from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


META_CANDIDATES = {
    "faultnumber",
    "fault_number",
    "fault",
    "simulationrun",
    "simulation_run",
    "run",
    "run_id",
    "sample",
    "time",
    "timestamp",
}

#reads Rdata files and extracts a pandas DataFrame

def _load_rdata(path: str) -> pd.DataFrame:
    try:
        import pyreadr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'pyreadr'. Install with: pip install pyreadr"
        ) from exc

    payload = pyreadr.read_r(path)
    if not payload:
        raise SystemExit(f"No objects found in {path}")
    for _, obj in payload.items():
        if isinstance(obj, pd.DataFrame):
            return obj
    raise SystemExit(f"No dataframe object found in {path}")


def _pick_col(columns: list[str], options: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in options:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _normalize_variable_name(name: str) -> str:
    raw = name.strip().lower().replace(" ", "").replace("-", "_")
    if raw.startswith("xmeas") and "_" not in raw:
        return f"xmeas_{raw.replace('xmeas', '')}"
    if raw.startswith("xmv") and "_" not in raw:
        return f"xmv_{raw.replace('xmv', '')}"
    return raw


def _load_variable_mapping(path: str) -> dict[str, dict]:
    df = pd.read_csv(path)
    required = {"variable", "instrument_type", "arr17_tag", "description", "unit"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Variable mapping file missing columns: {sorted(missing)}")

    mapping: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        var = _normalize_variable_name(str(row["variable"]))
        mapping[var] = {
            "variable": var,
            "instrument_type": str(row["instrument_type"]),
            "arr17_tag": str(row["arr17_tag"]),
            "description": str(row["description"]),
            "unit": str(row["unit"]),
        }
    return mapping


def _load_threshold_mapping(path: str) -> dict[str, dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    required = {"variable", "instrument_type", "arr17_tag", "hi_alarm", "lo_alarm"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Threshold mapping file missing columns: {sorted(missing)}")
    mapping: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        var = _normalize_variable_name(str(row["variable"]))
        mapping[var] = {
            "variable": var,
            "instrument_type": str(row["instrument_type"]),
            "arr17_tag": str(row["arr17_tag"]),
            "description": str(row.get("description", "")),
            "hi_alarm": float(row["hi_alarm"]) if pd.notna(row["hi_alarm"]) else None,
            "lo_alarm": float(row["lo_alarm"]) if pd.notna(row["lo_alarm"]) else None,
        }
    return mapping


def _top_variable_columns(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 8) -> list[str]:
    if not numeric_cols:
        return []
    return df[numeric_cols].var(numeric_only=True).sort_values(ascending=False).head(top_n).index.tolist()


def _summarize_group(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    if not numeric_cols:
        return "No numeric process variables available."
    top = _top_variable_columns(df, numeric_cols, top_n=8)
    means = df[top].mean(numeric_only=True).to_dict()
    mins = df[top].min(numeric_only=True).to_dict()
    maxs = df[top].max(numeric_only=True).to_dict()
    snippets = []
    for col in top:
        snippets.append(
            f"{col}: mean={means[col]:.3f}, min={mins[col]:.3f}, max={maxs[col]:.3f}"
        )
    return " | ".join(snippets)


def _extract_alarm_features(
    chunk: pd.DataFrame,
    threshold_mapping: dict[str, dict],
) -> tuple[list[str], dict[str, int], int]:
    if not threshold_mapping:
        return [], {}, 0
    col_by_norm = {_normalize_variable_name(col): col for col in chunk.columns}
    count_by_tag: dict[str, int] = {}
    for var, meta in threshold_mapping.items():
        col = col_by_norm.get(var)
        if col is None:
            continue
        series = chunk[col]
        hi = meta.get("hi_alarm")
        lo = meta.get("lo_alarm")
        if hi is not None:
            n_hi = int((series > hi).sum())
            if n_hi > 0:
                tag = f"AH_{meta['instrument_type']}_{meta['arr17_tag']}"
                count_by_tag[tag] = count_by_tag.get(tag, 0) + n_hi
        if lo is not None:
            n_lo = int((series < lo).sum())
            if n_lo > 0:
                tag = f"AL_{meta['instrument_type']}_{meta['arr17_tag']}"
                count_by_tag[tag] = count_by_tag.get(tag, 0) + n_lo
    tags = sorted(count_by_tag)
    total = int(sum(count_by_tag.values()))
    return tags, count_by_tag, total


def _build_docs(
    df: pd.DataFrame,
    source_name: str,
    default_fault_type: str,
    max_groups: int,
    variable_mapping: dict[str, dict],
    threshold_mapping: dict[str, dict],
) -> list[dict]:
    fault_col = _pick_col(df.columns.tolist(), ["faultnumber", "fault_number", "fault"])
    run_col = _pick_col(df.columns.tolist(), ["simulationrun", "simulation_run", "run", "run_id"])
    time_col = _pick_col(df.columns.tolist(), ["sample", "time", "timestamp"])

    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in META_CANDIDATES
    ]
    if fault_col:
        group_cols = [fault_col]
        if run_col:
            group_cols.append(run_col)
        grouped = list(df.groupby(group_cols, dropna=False))
    elif run_col:
        grouped = list(df.groupby(run_col, dropna=False))
    else:
        grouped = [("all_rows", df)]

    docs: list[dict] = []
    for idx, (key, chunk) in enumerate(grouped):
        if idx >= max_groups:
            break
        if not isinstance(key, tuple):
            key = (key,)

        fault_value = default_fault_type
        if fault_col:
            fault_value = str(key[0])
            if fault_value in {"0", "0.0"}:
                fault_value = "normal"

        run_value = None
        if run_col:
            run_value = str(key[-1]) if fault_col else str(key[0])

        time_scale = "minutes"
        if time_col:
            series = chunk[time_col].dropna()
            if not series.empty and series.max() <= 120:
                time_scale = "seconds"

        top_vars = _top_variable_columns(chunk, numeric_cols, top_n=8)
        mapped_signals = []
        alarm_tags = []
        for col in top_vars:
            mapped = variable_mapping.get(_normalize_variable_name(col))
            if not mapped:
                continue
            mapped_signals.append(mapped)
            alarm_tags.append(f"{mapped['instrument_type']}_{mapped['arr17_tag']}")

        threshold_tags, alarm_count_by_tag, alarm_count_total = _extract_alarm_features(
            chunk, threshold_mapping=threshold_mapping
        )
        if threshold_tags:
            alarm_tags = threshold_tags
        else:
            alarm_tags = sorted(set(alarm_tags))

        text = _summarize_group(chunk, numeric_cols)
        if mapped_signals:
            mapped_text = "; ".join(
                f"{item['variable']}={item['description']} ({item['instrument_type']}_{item['arr17_tag']})"
                for item in mapped_signals
            )
            text = f"{text} | mapped_signals: {mapped_text}"
        doc_id = f"{source_name}-{fault_value}-run-{run_value or idx}"
        docs.append(
            {
                "id": doc_id,
                "type": "fault_library" if fault_value != "normal" else "normal_operation",
                "title": f"{source_name} fault={fault_value} run={run_value or 'na'}",
                "text": text,
                "fault_type": fault_value,
                "alarm_tags": alarm_tags,
                "operating_region": "upset" if fault_value != "normal" else "normal",
                "time_scale": time_scale,
                "simulator_version": "dataverse",
                "source_dataset": source_name,
                "num_rows": int(len(chunk)),
                "top_variables": top_vars,
                "mapped_signals": mapped_signals,
                "alarm_count_total": alarm_count_total,
                "alarm_count_unique_tags": len(alarm_count_by_tag),
                "alarm_count_by_tag": alarm_count_by_tag,
            }
        )
    return docs


def _write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TE KB JSONL from faulty/fault-free RData")
    parser.add_argument("--faulty-rdata", required=True)
    parser.add_argument("--faultfree-rdata", required=True)
    parser.add_argument("--output", default="data/te_knowledge_base.generated.jsonl")
    parser.add_argument(
        "--variable-mapping-csv",
        default="data/te_variable_mapping.csv",
        help="CSV mapping for xmeas/xmv variables to Arr17 tags",
    )
    parser.add_argument(
        "--thresholds-csv",
        default="data/te_alarm_thresholds.csv",
        help="CSV of HI/LO thresholds used to generate alarm tags",
    )
    parser.add_argument("--max-groups-per-source", type=int, default=300, help="Caps documents per input source")
    args = parser.parse_args()

    faulty_df = _load_rdata(args.faulty_rdata)
    faultfree_df = _load_rdata(args.faultfree_rdata)
    variable_mapping = _load_variable_mapping(args.variable_mapping_csv)
    threshold_mapping = _load_threshold_mapping(args.thresholds_csv)

    docs_faulty = _build_docs(
        faulty_df,
        source_name="faulty_training",
        default_fault_type="faulty",
        max_groups=args.max_groups_per_source,
        variable_mapping=variable_mapping,
        threshold_mapping=threshold_mapping,
    )
    docs_normal = _build_docs(
        faultfree_df,
        source_name="faultfree_training",
        default_fault_type="normal",
        max_groups=args.max_groups_per_source,
        variable_mapping=variable_mapping,
        threshold_mapping=threshold_mapping,
    )
    docs = docs_faulty + docs_normal
    _write_jsonl(args.output, docs)
    print(f"Wrote {len(docs)} documents to {args.output}")


if __name__ == "__main__":
    main()
