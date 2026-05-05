import argparse
import re
import pandas as pd


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def is_true_value(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x == 1
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def filter_true_columns(df: pd.DataFrame, true_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in true_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required TRUE columns: {missing}")

    mask = pd.Series(True, index=df.index)
    for c in true_cols:
        mask &= df[c].apply(is_true_value)
    return df[mask].copy()


def infer_year(df: pd.DataFrame) -> pd.Series:
    for c in ["year", "publication_year", "pub_year"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    candidate_cols = [c for c in ["id", "title", "venue", "booktitle"] if c in df.columns]
    text = df[candidate_cols].fillna("").astype(str).agg(" ".join, axis=1) if candidate_cols else pd.Series([""] * len(df), index=df.index)
    years = text.str.extract(r"((?:19|20)\d{2})", expand=False)
    return pd.to_numeric(years, errors="coerce")


def infer_paper_group(df: pd.DataFrame) -> pd.Series:
    for c in ["paper_group", "track", "venue_type"]:
        if c in df.columns:
            return df[c].astype(str).str.lower()

    candidate_cols = [c for c in ["venue", "booktitle", "main_venue_match", "id", "title"] if c in df.columns]
    text = df[candidate_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if candidate_cols else pd.Series([""] * len(df), index=df.index)
    return text.apply(lambda x: "findings" if "findings" in x else "main")


def load_preselected_ids(preselect_file: str) -> list[str]:
    """Load preselected IDs from a txt (one per line) or xlsx/csv file."""
    if preselect_file.endswith(".xlsx"):
        pre_df = pd.read_excel(preselect_file)
        # take first column as IDs
        return pre_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    elif preselect_file.endswith(".csv"):
        pre_df = pd.read_csv(preselect_file)
        return pre_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    else:
        # plain txt: one id per line
        with open(preselect_file, "r") as f:
            return [line.strip() for line in f if line.strip()]


def sample_per_group(
    df: pd.DataFrame,
    group_cols: list[str],
    n: int = 20,
    seed: int = 42,
    preselected_ids: list[str] | None = None,
    id_col: str = "id",
    main_label: str = "main",
) -> pd.DataFrame:
    results = []

    for group_key, group_df in df.dropna(subset=group_cols).groupby(group_cols):
        year_val, group_label = group_key

        # For main papers: use preselected ids if provided
        if group_label == main_label and preselected_ids and id_col in df.columns:
            preselected = group_df[group_df[id_col].astype(str).str.strip().isin(preselected_ids)]
            remaining = group_df[~group_df[id_col].astype(str).str.strip().isin(preselected_ids)]

            n_pre = len(preselected)
            n_remaining = max(0, n - n_pre)

            sampled_remaining = remaining.sample(n=min(n_remaining, len(remaining)), random_state=seed) if n_remaining > 0 else remaining.iloc[0:0]
            combined = pd.concat([preselected, sampled_remaining]).head(n)

            print(
                f"  Group ({year_val}, {group_label}): "
                f"{n_pre} preselected + {len(sampled_remaining)} random = {len(combined)} total"
            )
            results.append(combined)
        else:
            k = min(n, len(group_df))
            results.append(group_df.sample(n=k, random_state=seed))

    return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input XLSX file path")
    parser.add_argument("--output", required=True, help="Output XLSX file path")
    parser.add_argument("--n", type=int, default=20, help="Samples per group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # TRUE filter columns
    parser.add_argument(
        "--true-cols",
        default="has_explanation_evaluation,is_xai_interpretable",
        help='Comma-separated columns that must be TRUE (default: "has_explanation_evaluation,is_xai_interpretable")',
    )
    parser.add_argument(
        "--extra-true-cols",
        default="",
        help="Additional comma-separated columns that must be TRUE",
    )

    # Preselection for main papers
    parser.add_argument(
        "--preselect-ids",
        default=None,
        help="Path to a .txt / .csv / .xlsx file containing preselected IDs for main papers (one ID per line for txt)",
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help='Column name used as paper ID (default: "id")',
    )
    parser.add_argument(
        "--main-label",
        default="main",
        help='Label used to identify main papers in paper_group (default: "main")',
    )

    args = parser.parse_args()

    df = pd.read_excel(args.input)

    true_cols = parse_list_arg(args.true_cols) + parse_list_arg(args.extra_true_cols)
    true_cols = list(dict.fromkeys(true_cols))

    if true_cols:
        df = filter_true_columns(df, true_cols)

    if "year" not in df.columns:
        df["year"] = infer_year(df)

    if "paper_group" not in df.columns:
        df["paper_group"] = infer_paper_group(df)

    # Load preselected IDs if provided
    preselected_ids = None
    if args.preselect_ids:
        preselected_ids = load_preselected_ids(args.preselect_ids)
        print(f"Loaded {len(preselected_ids)} preselected IDs from: {args.preselect_ids}")

    group_cols = ["year", "paper_group"]
    sampled = sample_per_group(
        df,
        group_cols=group_cols,
        n=args.n,
        seed=args.seed,
        preselected_ids=preselected_ids,
        id_col=args.id_col,
        main_label=args.main_label,
    )

    sampled.to_excel(args.output, index=False)

    print("\nSaved:", args.output)
    print("Selected rows after TRUE filter:", len(df))
    print(sampled.groupby(group_cols).size().rename("sampled_count"))


if __name__ == "__main__":
    main()