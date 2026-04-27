import argparse
import re
import pandas as pd


def infer_year(df: pd.DataFrame) -> pd.Series:
    # Prefer an existing year column if present
    for c in ["year", "publication_year", "pub_year"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    # Otherwise, try extracting year from common text columns (e.g., id/title)
    candidate_cols = [c for c in ["id", "title", "venue", "booktitle"] if c in df.columns]
    text = df[candidate_cols].fillna("").astype(str).agg(" ".join, axis=1) if candidate_cols else pd.Series([""] * len(df))
    years = text.str.extract(r"((?:19|20)\d{2})", expand=False)
    return pd.to_numeric(years, errors="coerce")


def infer_paper_group(df: pd.DataFrame) -> pd.Series:
    # If already present, use it
    for c in ["paper_group", "track", "venue_type"]:
        if c in df.columns:
            return df[c].astype(str).str.lower()

    # Infer from venue-like columns
    candidate_cols = [c for c in ["venue", "booktitle", "main_venue_match", "id", "title"] if c in df.columns]
    text = df[candidate_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if candidate_cols else pd.Series([""] * len(df))

    # Rule: contains "findings" => findings, else main
    return text.apply(lambda x: "findings" if "findings" in x else "main")


def sample_per_group(df: pd.DataFrame, group_cols, n=20, seed=42) -> pd.DataFrame:
    def _sample(g):
        k = min(n, len(g))
        return g.sample(n=k, random_state=seed)

    out = (
        df.dropna(subset=group_cols)
          .groupby(group_cols, group_keys=False)
          .apply(_sample)
          .reset_index(drop=True)
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input XLSX file path")
    parser.add_argument("--output", required=True, help="Output XLSX file path")
    parser.add_argument("--n", type=int, default=20, help="Samples per group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_excel(args.input)

    if "year" not in df.columns:
        df["year"] = infer_year(df)

    if "paper_group" not in df.columns:
        df["paper_group"] = infer_paper_group(df)

    group_cols = ["year", "paper_group"]
    sampled = sample_per_group(df, group_cols=group_cols, n=args.n, seed=args.seed)

    sampled.to_excel(args.output, index=False)

    # Optional summary
    print("Saved:", args.output)
    print(sampled.groupby(group_cols).size().rename("sampled_count"))


if __name__ == "__main__":
    main()