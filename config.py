"""Shared paths and constants.

The E-DAIC-WoZ corpus is not redistributed with this repository (see README).
Point RAW_DIR at your own copy, either by setting the EDAIC_RAW_DIR environment
variable or by placing the dataset under data/raw/.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset location on the company device.
COMPANY_DATASET_DIR = (
    Path.home()
    / "Desktop"
    / "depression project"
    / "Dataset"
    / "Depression Dataset"
)
LOCAL_DATASET_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RAW_DIR = (
    COMPANY_DATASET_DIR if COMPANY_DATASET_DIR.is_dir() else LOCAL_DATASET_DIR
)

# Directory holding the E-DAIC-WoZ release:
#   <RAW_DIR>/300_P.tar.gz, 301_P.tar.gz, ...
#   <RAW_DIR>/train_split.csv, dev_split.csv, test_split.csv
RAW_DIR = Path(
    os.environ.get(
        "EDAIC_RAW_DIR",
        os.environ.get("EDAIC_DATASET_PATH", DEFAULT_RAW_DIR),
    )
).expanduser()

# Generated feature tables land here (git-ignored).
PROCESSED_DIR = Path(os.environ.get("EDAIC_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))

# Metrics and figures (committed).
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42

# The 17 OpenFace Action Unit intensity columns used as the feature source.
AU_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
]

# Per-AU summary statistics: 17 AUs x 7 statistics = 119 features.
STATISTICS = ["mean", "std", "var", "min", "max", "skew", "kurt"]

# Columns that are metadata rather than model inputs.
META_COLS = ["Participant_ID", "Gender", "depressed_label", "PHQ_Score",
             "n_frames", "duration_s"]


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
