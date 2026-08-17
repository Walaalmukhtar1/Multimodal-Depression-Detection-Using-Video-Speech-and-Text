# -*- coding: utf-8 -*-
"""Step 2 — attach PHQ-8 labels to the pooled features.

Joins the pooled per-participant features against the dataset's official
train/dev/test split files, producing one labelled table per split.
PHQ_Binary (1 = PHQ-8 >= 10) is renamed to depressed_label.

Usage:
    python video/merge_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402

POOLED_CSV = PROCESSED_DIR / "au_pooled_features_all_participants.csv"
LABEL_COLS = ["Participant_ID", "Gender", "PHQ_Binary", "PHQ_Score"]
DROP_COLS = ["PCL-C (PTSD)", "PTSD Severity"]   # unrelated labels in the split files
SPLIT_FILES = {
    "train": ["train_split.csv", "Train Split Data.csv"],
    "dev": ["dev_split.csv", "Dev Split Data.csv"],
    "test": ["test_split.csv", "Test Split Data.csv"],
}


def merge_and_report(split_df, features, name):
    merged = split_df.merge(features, on="Participant_ID", how="left")

    missing = merged[merged["n_frames"].isna()]
    if len(missing):
        print(f"[{name}] WARNING: {len(missing)} participants have no pooled "
              f"features: {missing['Participant_ID'].tolist()}")
        merged = merged.dropna(subset=["n_frames"])

    feature_cols = [c for c in merged.columns
                    if c not in LABEL_COLS and c not in DROP_COLS]
    merged = merged[LABEL_COLS + feature_cols]
    merged = merged.rename(columns={"PHQ_Binary": "depressed_label"})

    n_dep = int(merged["depressed_label"].sum())
    print(f"[{name}] {len(merged)} participants | depressed={n_dep} "
          f"not_depressed={len(merged) - n_dep}")
    return merged


def main():
    ensure_dirs()
    if not POOLED_CSV.exists():
        sys.exit(f"{POOLED_CSV} not found — run video/pool_features.py first.")

    features = pd.read_csv(POOLED_CSV)
    features["Participant_ID"] = features["Participant_ID"].astype(int)

    for name, file_names in SPLIT_FILES.items():
        split_path = next(
            (RAW_DIR / file_name for file_name in file_names
             if (RAW_DIR / file_name).is_file()),
            None,
        )
        if split_path is None:
            tried = ", ".join(file_names)
            sys.exit(f"No {name} label file found in {RAW_DIR}. Tried: {tried}")
        out = merge_and_report(pd.read_csv(split_path), features, name)
        out_path = PROCESSED_DIR / f"au_features_{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"         -> {out_path}")


if __name__ == "__main__":
    main()
