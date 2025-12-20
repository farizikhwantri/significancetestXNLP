import argparse
import os
import pandas as pd


def detect_boolean_columns(df: pd.DataFrame) -> list[str]:
    # Select columns where all non-null values are strictly boolean
    bool_cols = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if series.map(lambda v: isinstance(v, bool)).all():
            bool_cols.append(col)
    return sorted(bool_cols)


def write_csv(jsonl_path: str, csv_path: str) -> None:
    df = pd.read_json(jsonl_path, lines=True)

    # Keep only id, title, and boolean columns
    bool_cols = detect_boolean_columns(df)
    cols = [c for c in ["id", "title"] if c in df.columns] + bool_cols
    out = df[cols].copy()

    # Map booleans to lowercase strings; leave blanks for missing
    for col in bool_cols:
        out[col] = out[col].map({True: "true", False: "false"})
        out[col] = out[col].fillna("")

    # Ensure directory exists and write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to CSV with id, title, and top-level boolean columns (skip *_raw)."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="/Users/fariz/repositories/significancetestXNLP/output/xai_llm_results.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/Users/fariz/repositories/significancetestXNLP/output/xai_llm_results.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()
    write_csv(args.input, args.output)


if __name__ == "__main__":
    main()