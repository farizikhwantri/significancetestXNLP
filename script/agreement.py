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
    if s == "nan" or s == "":
        return np.nan
    return np.nan


def resolve_column(columns_norm: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns_norm:
            return columns_norm[c]
    return None


def krippendorff_alpha_binary(r1: np.ndarray, r2: np.ndarray) -> float:
    do = np.mean(r1 != r2)
    pooled = np.concatenate([r1, r2])
    p1 = np.mean(pooled == 1)
    p0 = 1.0 - p1
    de = 1.0 - (p0**2 + p1**2)
    if de == 0:
        return 1.0 if do == 0 else np.nan
    return 1.0 - (do / de)


def scotts_pi(a: np.ndarray, b: np.ndarray) -> float:
    """Scott's Pi - uses pooled marginals, better for binary when base rates differ."""
    pooled = np.concatenate([a, b])
    p1 = np.mean(pooled == 1)
    p0 = 1.0 - p1
    p_exp = p1**2 + p0**2
    p_obs = np.mean(a == b)
    if p_exp == 1.0:
        return 1.0 if p_obs == 1.0 else np.nan
    return (p_obs - p_exp) / (1.0 - p_exp)


def pabak(a: np.ndarray, b: np.ndarray) -> float:
    """PABAK - Prevalence-Adjusted Bias-Adjusted Kappa.
    Corrects for kappa paradox when prevalence is skewed (mostly TRUE or mostly FALSE).
    Formula: 2 * p_obs - 1
    Interpretation: same scale as kappa but ignores prevalence imbalance.
    """
    p_obs = np.mean(a == b)
    return 2 * p_obs - 1


def compute_metrics(a: pd.Series, b: pd.Series) -> Tuple[int, float, float, float, float, float]:
    mask = a.notna() & b.notna()
    a2 = a[mask].astype(int).to_numpy()
    b2 = b[mask].astype(int).to_numpy()
    n = len(a2)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    p_obs = np.mean(a2 == b2)
    kappa = cohen_kappa_score(a2, b2)
    alpha = krippendorff_alpha_binary(a2, b2)
    pi = scotts_pi(a2, b2)
    pk = pabak(a2, b2)
    return n, round(p_obs, 6), round(kappa, 6), round(pi, 6), round(pk, 6), round(alpha, 6)


def check_duplicates(df: pd.DataFrame, id_col: str, source_name: str):
    """Error if duplicate IDs exist, reporting row numbers (1-based, header excluded)."""
    dupes = df[df[id_col].duplicated(keep=False)]
    if not dupes.empty:
        for id_val, group in dupes.groupby(id_col):
            rows = [str(i + 2) for i in group.index.tolist()]
            print(f"[ERROR] Duplicate ID '{id_val}' in {source_name} at row(s): {', '.join(rows)}", file=sys.stderr)
        sys.exit(1)


COLUMNS = ["has_explanation_evaluation", "is_xai_interpretable", "has_significance_testing"]

COL_ALIASES = {
    "has_explanation_evaluation": [
        "has_explanation_evaluation",
        "has_explanalation_evaluation",
        "has_explanation_evaluation_",
        "has_explanalation_evaluation_",
    ],
    "is_xai_interpretable": [
        "is_xai_interpretable",
    ],
    "has_significance_testing": [
        "has_significance_testing",
        "has_significance_testing_",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Binary IAA (Cohen's kappa, PABAK, Scott's Pi, Krippendorff's alpha) between CSV and XLSX."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file (annotator A)")
    parser.add_argument("--xlsx", required=True, help="Path to XLSX file (annotator B)")
    parser.add_argument("--sheet", required=True, help="Sheet name in XLSX")
    parser.add_argument("--id-col", default="id", help="ID column used to match rows (default: id)")
    parser.add_argument("--verified-col", default="verified", help="Verified column in XLSX (default: verified)")
    parser.add_argument("--out", default="", help="Optional output CSV for results")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    df_csv = pd.read_csv(args.csv, dtype=str)
    df_xlsx = pd.read_excel(args.xlsx, sheet_name=args.sheet, dtype=str)

    csv_cols_norm = {norm_col(c): c for c in df_csv.columns}
    xlsx_cols_norm = {norm_col(c): c for c in df_xlsx.columns}

    # ── Filter XLSX by verified == TRUE ───────────────────────────────────────
    verified_norm = norm_col(args.verified_col)
    verified_xlsx = xlsx_cols_norm.get(verified_norm)
    if not verified_xlsx:
        print(f"[ERROR] Verified column '{args.verified_col}' not found in XLSX.", file=sys.stderr)
        sys.exit(1)

    df_xlsx = df_xlsx[df_xlsx[verified_xlsx].astype(str).str.strip().str.upper() == "TRUE"].copy()
    df_xlsx = df_xlsx.reset_index(drop=False)
    print(f"[INFO] XLSX rows after {verified_xlsx} == TRUE filter: {len(df_xlsx)}")

    # ── Resolve ID column ─────────────────────────────────────────────────────
    id_norm = norm_col(args.id_col)
    id_csv = csv_cols_norm.get(id_norm)
    id_xlsx = xlsx_cols_norm.get(id_norm)

    if not id_csv or not id_xlsx:
        print(f"[ERROR] ID column '{args.id_col}' not found in both files.", file=sys.stderr)
        sys.exit(1)

    # ── Normalize IDs ─────────────────────────────────────────────────────────
    df_csv[id_csv] = df_csv[id_csv].astype(str).str.strip().str.lower()
    df_xlsx[id_xlsx] = df_xlsx[id_xlsx].astype(str).str.strip().str.lower()

    # Drop empty IDs (blank trailing rows)
    df_csv = df_csv[df_csv[id_csv].notna() & (df_csv[id_csv] != "") & (df_csv[id_csv] != "nan")].copy()
    df_xlsx = df_xlsx[df_xlsx[id_xlsx].notna() & (df_xlsx[id_xlsx] != "") & (df_xlsx[id_xlsx] != "nan")].copy()

    # ── Duplicate check ───────────────────────────────────────────────────────
    check_duplicates(df_csv, id_csv, "CSV")
    check_duplicates(df_xlsx, id_xlsx, "XLSX")

    # ── Resolve annotation columns ────────────────────────────────────────────
    resolved = {}
    for key, candidates in COL_ALIASES.items():
        c_csv = resolve_column(csv_cols_norm, candidates)
        c_xlsx = resolve_column(xlsx_cols_norm, candidates)
        if not c_csv or not c_xlsx:
            print(f"[ERROR] Could not find column '{key}' in both files.", file=sys.stderr)
            sys.exit(1)
        resolved[key] = (c_csv, c_xlsx)

    # ── Match rows by ID ──────────────────────────────────────────────────────
    xlsx_lookup = df_xlsx.set_index(id_xlsx)

    records = []
    unmatched_csv = []

    for _, csv_row in df_csv.iterrows():
        row_id = csv_row[id_csv]
        if row_id not in xlsx_lookup.index:
            unmatched_csv.append(row_id)
            continue
        xlsx_row = xlsx_lookup.loc[row_id]
        record = {"id": row_id}
        for key, (c_csv, c_xlsx) in resolved.items():
            record[f"{key}_csv"] = normalize_bool(csv_row[c_csv])
            record[f"{key}_xlsx"] = normalize_bool(xlsx_row[c_xlsx])
        records.append(record)

    if unmatched_csv:
        print(f"[WARN] {len(unmatched_csv)} CSV IDs not found in XLSX:")
        for uid in unmatched_csv:
            print(f"  - {uid}")

    unmatched_xlsx = [i for i in xlsx_lookup.index if i not in df_csv[id_csv].values]
    if unmatched_xlsx:
        print(f"[WARN] {len(unmatched_xlsx)} XLSX IDs not found in CSV:")
        for uid in unmatched_xlsx:
            print(f"  - {uid}")

    matched = pd.DataFrame(records)
    print(f"[INFO] Matched rows: {len(matched)}")

    if matched.empty:
        print("[ERROR] No matching rows found. Check ID column values.", file=sys.stderr)
        sys.exit(1)

    # ── Per-column analysis ───────────────────────────────────────────────────
    result_rows = []
    for base in COLUMNS:
        csv_col = matched[f"{base}_csv"]
        xlsx_col = matched[f"{base}_xlsx"]

        mask = csv_col.notna() & xlsx_col.notna()
        a2 = csv_col[mask].astype(int)
        b2 = xlsx_col[mask].astype(int)
        total = len(a2)

        p_csv_1 = (a2 == 1).mean()
        p_xlsx_1 = (b2 == 1).mean()
        p_csv_0 = 1 - p_csv_1
        p_xlsx_0 = 1 - p_xlsx_1
        p_agree_obs = (a2 == b2).mean()
        p_agree_exp = (p_csv_1 * p_xlsx_1) + (p_csv_0 * p_xlsx_0)

        print(f"\n{'='*60}")
        print(f"[COLUMN] {base}")
        print(f"  CSV  TRUE rate : {p_csv_1:.3f}  ({int(p_csv_1*total)}/{total})")
        print(f"  XLSX TRUE rate : {p_xlsx_1:.3f}  ({int(p_xlsx_1*total)}/{total})")
        print(f"  Observed agreement            : {p_agree_obs:.3f} ({int(p_agree_obs*total)}/{total})")
        print(f"  Expected agreement (by chance): {p_agree_exp:.3f}")
        print(f"  => Kappa = (obs-exp)/(1-exp) = ({p_agree_obs:.3f}-{p_agree_exp:.3f})/(1-{p_agree_exp:.3f})")
        print(f"  => PABAK = 2*obs-1 = 2*{p_agree_obs:.3f}-1 (ignores prevalence skew)")

        ct = pd.crosstab(a2, b2, rownames=["CSV"], colnames=["XLSX"])
        print(f"\n  Confusion matrix:\n{ct}")

        # Disagreements
        disagree_mask = a2 != b2
        disagree_df = matched.loc[disagree_mask[disagree_mask].index, ["id", f"{base}_csv", f"{base}_xlsx"]]
        if disagree_df.empty:
            print(f"\n  No disagreements!")
        else:
            print(f"\n  Disagreements ({len(disagree_df)}):")
            print(disagree_df.to_string(index=False))

        n, p_obs, kappa, pi, pk, alpha = compute_metrics(csv_col, xlsx_col)
        result_rows.append({
            "column": base,
            "n": n,
            "observed_agreement": p_obs,
            "cohen_kappa": kappa,
            "scotts_pi": pi,
            "pabak": pk,
            "krippendorff_alpha": alpha,
        })

    # ── Per-row agreement ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[PER-ROW] Agreement per paper across all columns:")
    for _, row in matched.iterrows():
        row_results = []
        for base in COLUMNS:
            csv_val = row[f"{base}_csv"]
            xlsx_val = row[f"{base}_xlsx"]
            if pd.notna(csv_val) and pd.notna(xlsx_val):
                match = "✔" if csv_val == xlsx_val else f"✘(csv={int(csv_val)} xlsx={int(xlsx_val)})"
                row_results.append(f"{base}: {match}")
        print(f"  {row['id']}: {' | '.join(row_results)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    result = pd.DataFrame(result_rows)
    print(f"\n{'='*60}")
    print("[RESULTS]")
    print(f"\n  NOTE: Cohen's kappa can be 0 or negative when label prevalence is heavily")
    print(f"  skewed (e.g. mostly TRUE or mostly FALSE). In such cases, PABAK is the")
    print(f"  recommended metric as it adjusts for prevalence imbalance.")
    print(f"  Interpretation: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print(f"                  0.61-0.80 substantial, >0.80 almost perfect\n")
    print(result.to_string(index=False))

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()