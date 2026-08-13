# -*- coding: utf-8 -*-
"""Step 1 — pool per-frame OpenFace Action Units into per-participant features.

For every participant archive in RAW_DIR, extracts the OpenFace CSV, keeps only
successfully tracked frames (success == 1), and reduces each of the 17 Action
Unit intensity signals to 7 summary statistics, giving 119 features per person.

Usage:
    python video/pool_features.py           # all participants
    python video/pool_features.py 10        # first 10, for a quick check
"""
import glob
import os
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, AU_COLS, ensure_dirs  # noqa: E402

OUT_CSV = PROCESSED_DIR / "au_pooled_features_all_participants.csv"
ERR_LOG = PROCESSED_DIR / "pool_features_errors.log"


def stats_row(pid, df):
    """Reduce one participant's frame-level AU signals to summary statistics."""
    row = {
        "Participant_ID": pid,
        "n_frames": len(df),
        "duration_s": round(df["timestamp"].max(), 2),
    }
    for au in AU_COLS:
        col = df[au]
        row[f"{au}_mean"] = col.mean()
        row[f"{au}_std"] = col.std()
        row[f"{au}_var"] = col.var()
        row[f"{au}_min"] = col.min()
        row[f"{au}_max"] = col.max()
        row[f"{au}_skew"] = col.skew()
        row[f"{au}_kurt"] = col.kurt()
    return row


def process_one(path):
    pid = os.path.basename(path).split("_")[0]
    member = f"{pid}_P/features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    t0 = time.time()
    with tarfile.open(path, "r:gz") as tf:
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(member)
        df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["success"] == 1]          # drop frames where tracking failed
    return pid, stats_row(pid, df), time.time() - t0


def main():
    ensure_dirs()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    archives = sorted(glob.glob(str(RAW_DIR / "*_P.tar.gz")))
    if not archives:
        sys.exit(f"No *_P.tar.gz archives found in {RAW_DIR}.\n"
                 f"Set EDAIC_RAW_DIR or place the dataset under data/raw/ "
                 f"(see README).")
    if limit:
        archives = archives[:limit]
    print(f"Found {len(archives)} archives to process", flush=True)

    rows, errors = [], []
    for i, path in enumerate(archives, 1):
        try:
            pid, row, dt = process_one(path)
            rows.append(row)
            print(f"[{i}/{len(archives)}] {pid} ok in {dt:.1f}s", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(archives)}] {os.path.basename(path)} FAILED: {exc}", flush=True)
            errors.append(f"{os.path.basename(path)}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} participant rows to {OUT_CSV}")
    if errors:
        ERR_LOG.write_text("\n".join(errors))
        print(f"{len(errors)} errors logged to {ERR_LOG}")


if __name__ == "__main__":
    main()
