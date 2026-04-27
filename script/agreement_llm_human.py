#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_bool(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()

    true_vals = {"1", "true", "t", "yes", "y", "✔", "check", "checked"}
    false_vals = {"0", "false", "f", "no", "n", "✘", "x"}

    if s in true_vals:
        return 1
    if s in false_vals:
        return 0

    # handle bool-like python values
    if s == "nan" or s == "":
        return np.nan

    return np.nan


def resolve_column(columns_norm: Dict[str, str], candidates: List[str]) -> Optional[str]:
    """
    columns_norm: {normalized_name: original_name}
    candidates: list of possible normalized names
    returns original column name if found
    """
    for c in candidates:
        if c in columns_norm:
            return columns_norm[c]
    return None


def krippendorff_alpha_binary(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Binary nominal alpha for two raters, no missing values.
    """
    do = np.mean(r1 != r2)  # observed disagreement
    pooled = np.concatenate([r1, r2])
    p1 = np.mean(pooled == 1)
    p0 = 1.0 - p1
    de = 1.0 - (p0**2 + p1**2)  # expected disagreement = 2*p0*p1
    if de == 0:
        return 1.0 if do == 0 else np.nan
    return 1.0 - (do / de)


def compute_metrics(a: pd.Series, b: pd.Series) -> Tuple[int, float, float]:
    mask = a.notna() & b.notna()
    a2 = a[mask].astype(int).to_numpy()
    b2 = b[mask].astype(int).to_numpy()
    n = len(a2)
    if n == 0:
        return 0, np.nan, np.nan

    kappa = cohen_kappa_score(a2, b2)
    alpha = krippendorff_alpha_binary(a2, b2)
    return n, kappa, alpha


def main():
    parser = argparse.ArgumentParser(
        description="Binary agreement (Cohen's kappa, Krippendorff's alpha) between CSV and XLSX."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--xlsx", required=True, help="Path to XLSX file")
    parser.add_argument("--sheet", required=True, help="Sheet name in XLSX")
    parser.add_argument("--id-col", default="id", help="ID column used to align rows (default: id)")
    parser.add_argument("--verified-col", default="verified", help="Verified column in XLSX (default: verified)")
    parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV for results"
    )
    args = parser.parse_args()

    # Load files
    # make read only 
    df_csv = pd.read_csv(args.csv, dtype=str)
    df_xlsx = pd.read_excel(args.xlsx, sheet_name=args.sheet, dtype=str)

    # Build normalized column maps
    csv_cols_norm = {norm_col(c): c for c in df_csv.columns}
    xlsx_cols_norm = {norm_col(c): c for c in df_xlsx.columns}

    # # Resolve ID column in both files
    # id_norm = norm_col(args.id_col)
    # id_csv = csv_cols_norm.get(id_norm)
    # id_xlsx = xlsx_cols_norm.get(id_norm)

    # Filter XLSX: keep only verified == TRUE
    verified_norm = norm_col(args.verified_col)
    verified_xlsx = xlsx_cols_norm.get(verified_norm)
    if not verified_xlsx:
        print(f"[ERROR] Verified column '{args.verified_col}' not found in XLSX.", file=sys.stderr)
        sys.exit(1)

    df_xlsx = df_xlsx[df_xlsx[verified_xlsx].astype(str).str.strip().str.upper() == "TRUE"].copy()
    print(f"[INFO] XLSX rows after {verified_xlsx} == TRUE filter: {len(df_xlsx)}")

    # Resolve ID column in both files
    id_norm = norm_col(args.id_col)
    id_csv = csv_cols_norm.get(id_norm)
    id_xlsx = xlsx_cols_norm.get(id_norm)

    if not id_csv or not id_xlsx:
        print(f"[ERROR] ID column '{args.id_col}' not found in both files.", file=sys.stderr)
        sys.exit(1)

    # Column aliases (handles misspelling: has_explanalation_evaluation)
    col_aliases = {
        "has_explanation_evaluation": [
            "has_explanation_evaluation",
            "has_explanalation_evaluation",  # misspelling support
            "has_explanation_evaluation_",
            "has_explanalation_evaluation_",
            "has_explanation_evaluation",
            "has_explanalation_evaluation",
            "has_explanalation_evaluation",
            "has_explanalation_evaluation",
        ],
        "is_xai_interpretable": [
            "is_xai_interpretable",
        ],
    }

    resolved = {}
    for key, candidates in col_aliases.items():
        c_csv = resolve_column(csv_cols_norm, candidates)
        c_xlsx = resolve_column(xlsx_cols_norm, candidates)
        if not c_csv or not c_xlsx:
            print(f"[ERROR] Could not find column '{key}' in both files.", file=sys.stderr)
            sys.exit(1)
        resolved[key] = (c_csv, c_xlsx)

    # Subset + rename for merge
    csv_keep = [id_csv] + [resolved[k][0] for k in resolved]
    xlsx_keep = [id_xlsx] + [resolved[k][1] for k in resolved]

    a = df_csv[csv_keep].copy()
    b = df_xlsx[xlsx_keep].copy()

    a = a.rename(columns={id_csv: "id", resolved["has_explanation_evaluation"][0]: "has_explanation_evaluation_csv",
                          resolved["is_xai_interpretable"][0]: "is_xai_interpretable_csv"})
    b = b.rename(columns={id_xlsx: "id", resolved["has_explanation_evaluation"][1]: "has_explanation_evaluation_xlsx",
                          resolved["is_xai_interpretable"][1]: "is_xai_interpretable_xlsx"})

    merged = a.merge(b, on="id", how="inner")

    # Normalize binary labels
    for c in [
        "has_explanation_evaluation_csv",
        "has_explanation_evaluation_xlsx",
        "is_xai_interpretable_csv",
        "is_xai_interpretable_xlsx",
    ]:
        merged[c] = merged[c].map(normalize_bool)

    # Per-column metrics
    rows = []
    for base in ["has_explanation_evaluation", "is_xai_interpretable"]:
        n, kappa, alpha = compute_metrics(merged[f"{base}_csv"], merged[f"{base}_xlsx"])
        rows.append({"column": base, "n": n, "cohen_kappa": kappa, "krippendorff_alpha": alpha})

    # Overall metrics (stack both columns)
    overall_csv = pd.concat(
        [merged["has_explanation_evaluation_csv"], merged["is_xai_interpretable_csv"]],
        ignore_index=True,
    )
    overall_xlsx = pd.concat(
        [merged["has_explanation_evaluation_xlsx"], merged["is_xai_interpretable_xlsx"]],
        ignore_index=True,
    )
    n, kappa, alpha = compute_metrics(overall_csv, overall_xlsx)
    rows.append({"column": "OVERALL", "n": n, "cohen_kappa": kappa, "krippendorff_alpha": alpha})

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()