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


def _summarize_group(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    if not numeric_cols:
        return "No numeric process variables available."
    top = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False).head(8).index.tolist()
    means = df[top].mean(numeric_only=True).to_dict()
    mins = df[top].min(numeric_only=True).to_dict()
    maxs = df[top].max(numeric_only=True).to_dict()
    snippets = []
    for col in top:
        snippets.append(
            f"{col}: mean={means[col]:.3f}, min={mins[col]:.3f}, max={maxs[col]:.3f}"
        )
    return " | ".join(snippets)


def _build_docs(
    df: pd.DataFrame,
    source_name: str,
    default_fault_type: str,
    max_groups: int,
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

        text = _summarize_group(chunk, numeric_cols)
        doc_id = f"{source_name}-{fault_value}-run-{run_value or idx}"
        docs.append(
            {
                "id": doc_id,
                "type": "fault_library" if fault_value != "normal" else "normal_operation",
                "title": f"{source_name} fault={fault_value} run={run_value or 'na'}",
                "text": text,
                "fault_type": fault_value,
                "alarm_tags": [],
                "operating_region": "upset" if fault_value != "normal" else "normal",
                "time_scale": time_scale,
                "simulator_version": "dataverse",
                "source_dataset": source_name,
                "num_rows": int(len(chunk)),
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
    parser.add_argument("--max-groups-per-source", type=int, default=300, help="Caps documents per input source")
    args = parser.parse_args()

    faulty_df = _load_rdata(args.faulty_rdata)
    faultfree_df = _load_rdata(args.faultfree_rdata)

    docs_faulty = _build_docs(
        faulty_df, source_name="faulty_training", default_fault_type="faulty", max_groups=args.max_groups_per_source
    )
    docs_normal = _build_docs(
        faultfree_df, source_name="faultfree_training", default_fault_type="normal", max_groups=args.max_groups_per_source
    )
    docs = docs_faulty + docs_normal
    _write_jsonl(args.output, docs)
    print(f"Wrote {len(docs)} documents to {args.output}")


if __name__ == "__main__":
    main()
