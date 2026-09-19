"""
CSV batch runner for the "drop a CSV in input/, get a CSV in output/" front end.

Delegates all scoring to score_upload.score_patient_upload() -- the canonical
serving path (two-stage guard, encoding, SHAP) -- so this module only has to
find the uploaded CSV and flatten the returned records into output/result.csv.

Run:  python predict.py [--input PATH] [--output PATH]
Defaults: the single *.csv in input/, and output/result.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from score_upload import score_patient_upload

HERE = Path(__file__).resolve().parent
TOP_N_FACTORS = 3


def find_input_csv(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            sys.exit(f"Input file not found: {path}")
        return path
    input_dir = HERE / "input"
    candidates = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"No CSV file found in {input_dir}")
    if len(candidates) > 1:
        print(f"Multiple CSVs in {input_dir}; scoring the most recently modified: {candidates[0].name}")
    return candidates[0]


def flatten(records: list[dict]) -> list[dict]:
    """Turn score_upload's nested records (top_factors as a list of dicts)
    into flat columns a CSV can hold."""
    rows = []
    for rec in records:
        row = {
            "patient_id": rec.get("patient_id") or "",
            "age": rec.get("age") if rec.get("age") is not None else "",
            "condition": rec.get("condition") or "",
            "medication_name": rec.get("medication_name") or "",
            "risk_score": rec.get("risk_score") if rec.get("risk_score") is not None else "",
            "risk_tier": rec.get("risk_tier") or "",
            "reason": rec.get("reason") or "",
        }
        factors = rec.get("top_factors") or []
        for slot in range(1, TOP_N_FACTORS + 1):
            factor = factors[slot - 1] if slot <= len(factors) else None
            row[f"top_factor_{slot}"] = factor["factor"] if factor else ""
            row[f"top_factor_{slot}_impact"] = factor["impact"] if factor else ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV to score (default: the CSV found in input/)")
    parser.add_argument("--output", default=str(HERE / "output" / "result.csv"))
    args = parser.parse_args()

    input_path = find_input_csv(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {input_path} -> {output_path}")
    result = score_patient_upload(str(input_path))

    if result["status"] == "error":
        sys.exit(f"score_patient_upload failed: {result['error']}")

    print(f"total={result['total_patients']} scored={result['scored_count']} "
          f"insufficient_data={result['insufficient_data_count']} "
          f"high_risk={result['high_risk_count']}")

    pd.DataFrame(flatten(result["all_patients"])).to_csv(output_path, index=False)
    print(f"Wrote {result['total_patients']} rows -> {output_path}")


if __name__ == "__main__":
    main()
