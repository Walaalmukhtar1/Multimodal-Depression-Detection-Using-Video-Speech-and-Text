# Multimodal Depression Detection (E-DAIC-WoZ)

Binary depression detection (PHQ-8 ≥ 10) from clinical interview recordings,
using the E-DAIC-WoZ corpus of 275 participants.

The project is organised by modality. Each is developed independently against a
shared evaluation protocol, with a fusion stage.

---

## Video pipeline

Frame-level OpenFace Action Unit intensities are pooled into per-participant
summary statistics, then classified with logistic regression.

```
pool_features.py  ──►  merge_labels.py  ──►  train_logreg.py
 17 AUs × 7 stats       attach PHQ-8          train + evaluate
 = 119 features         labels by split
```

**Features.** Each of the 17 OpenFace AU intensity signals (`AU01_r` … `AU45_r`)
is reduced to 7 statistics — mean, standard deviation, variance, min, max,
skewness, kurtosis — giving 119 features per participant. Only frames where
OpenFace reported successful tracking (`success == 1`) are used.

**Model.** Logistic regression, `C=1.0`, L2 penalty, `class_weight='balanced'`
to handle the roughly 3:1 class imbalance. No grid search, no feature
selection, no dimensionality reduction — these were evaluated and none improved
on the plain model.

**Protocol.** Trained on train + dev combined (219 participants) and evaluated
once on the official held-out test set (56 participants). Features are
standardised with a scaler fit on training data only.

### Results

Held-out test set, 56 participants:

| Metric | Value |
|---|---|
| Accuracy | 0.661 |
| **Macro F1** | **0.626** |
| ROC-AUC | 0.609 |
| Macro precision | 0.625 |
| Macro recall | 0.640 |

---

## Text pipeline

Ten transcript features are extracted for each participant: average sentiment,
speech speed, unique-word frequency, stop-word frequency, average characters,
noun, verb, adjective and adverb frequencies, and first-person pronoun
frequency. A linear SVM (`C=0.01`) is evaluated with LOSO on the balanced
66/66 participant set.

Result: **58.33% accuracy, 0.636 depressed F1, 0.575 macro F1**.

## Audio pipeline

Participant-level high-level statistics are calculated from openSMILE eGeMAPS
frames. SelectKBest (`k=50`) is fitted inside each LOSO training fold, followed
by a linear SVM (`C=0.1`) on the same balanced participant set.

Result: **63.64% accuracy, 0.642 depressed F1, 0.636 macro F1**.

---

## Class distribution

| Split | Total | Depressed | Not depressed |
|---|---|---|---|
| Train | 163 | 37 | 126 |
| Dev | 56 | 12 | 44 |
| Train + dev (used for training) | 219 | 49 | 170 |
| Test (held out) | 56 | 17 | 39 |
| **Full corpus** | **275** | **66** | **209** |

---

## Setup

```bash
pip install -r requirements.txt
```

The E-DAIC-WoZ corpus is released under a data use agreement and is **not**
included in this repository. On the company device, the scripts automatically
use this location when it exists:

```text
~/Desktop/depression project/Dataset/Depression Dataset
```

On another device, either place the dataset under `data/raw/` or point the
scripts at your copy:

```bash
export EDAIC_RAW_DIR=/path/to/e-daic-woz
```

`EDAIC_RAW_DIR` always overrides the automatic company-device path.

Confirm the path before running the experiments:

```bash
python3 -c "from config import RAW_DIR; print(RAW_DIR)"
```

On the company device, the printed absolute path should end with:

```text
Desktop/depression project/Dataset/Depression Dataset
```

See [`data/README.md`](data/README.md) for the expected layout.

## Running

```bash
python video/pool_features.py     # ~275 archives, slow on first run
python video/merge_labels.py
python video/train_logreg.py

# Final balanced text and audio systems
python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger_eng
python text/edaic_nlp_system.py
python audio/edaic_audio_system.py
```

Generated feature tables are written to `data/processed/` (git-ignored, since
they contain participant identifiers and PHQ-8 scores). Metrics and figures are
written to `results/`.

## Repository layout

```
config.py           shared paths, seeds and column definitions
video/              facial Action Unit pipeline
audio/              speech pipeline
text/               transcript pipeline
fusion/             multimodal combination
data/               dataset and generated tables (git-ignored)
results/            metrics and figures
```

## Data statement

The corpus contains clinical assessment scores from human participants. No raw
recordings, participant identifiers, or PHQ-8 scores are committed to this
repository. Only aggregate model performance metrics are published here.
