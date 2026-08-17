# Text–Audio Early Fusion

This folder contains the text–audio early-fusion system for depression
classification using the E-DAIC dataset.

## Method

The experiment uses the same balanced group of 132 participants:

- 66 depressed participants
- 66 non-depressed participants

The files are matched using `Participant_ID` so that every participant's text
features, audio features, and depression label belong to the same person.

The system combines:

- All 10 transcript features
- The 50 best participant-level eGeMAPS audio features

The audio features are selected with `VarianceThreshold` and `SelectKBest`
using the ANOVA F-score. Feature selection is fitted only on the training
participants in each fold to prevent data leakage.

The 10 text features and 50 selected audio features are joined before model
training. This is feature-level **early fusion**.

## Transcript features

1. Average sentiment
2. Speech speed
3. Unique-word frequency
4. Stop-word frequency
5. Average characters per word
6. Noun frequency
7. Verb frequency
8. Adjective frequency
9. Adverb frequency
10. First-person pronoun frequency

## Model and evaluation

The combined features are standardized using `StandardScaler` and classified
with a linear SVM:

```python
SVC(kernel="linear", C=0.01)
```

Leave-One-Subject-Out (LOSO) evaluation is used. In every fold, one participant
is used for testing and the remaining 131 participants are used for training.
This is repeated until every participant has been tested once.

## Results

| Modality | Model | Accuracy | Depressed recall | Depressed precision | Depressed F1 | Non-depressed F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Text + eGeMAPS audio | Linear SVM | 64.39% | 0.606 | 0.656 | 0.630 | 0.657 | 0.643 |

Confusion matrix:

```text
[[45, 21],
 [26, 40]]
```

- 45 non-depressed participants were classified correctly.
- 21 non-depressed participants were classified as depressed.
- 26 depressed participants were classified as non-depressed.
- 40 depressed participants were classified correctly.

## Required processed files

The code expects these files inside `data/processed/`:

```text
edaic_nlp_balanced_loso_results.csv
edaic_nlp_features.csv
audio/high_level_statistics/edaic_egemaps_high_stats_features.csv
```

The E-DAIC dataset and participant-level processed tables are excluded from
GitHub because the dataset is distributed under a data-use agreement.

## Run the experiment

From the repository root, run:

```bash
python3 fusion/text_audio_early_fusion.py
```

The script prints the evaluation and saves it to:

```text
data/processed/fusion/text_audio_fusion_evaluation.csv
```

## Main file

- `text_audio_early_fusion.py` — loads the balanced data, selects the audio
  features inside each LOSO fold, performs early fusion, trains the linear SVM,
  and reports the final evaluation.
