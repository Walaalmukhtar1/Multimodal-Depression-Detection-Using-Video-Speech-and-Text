# Text-audio fusion

This pipeline performs feature-level fusion on the same balanced 132
participants used by the separate text and audio systems.

For every leave-one-subject-out fold:

1. The test participant is held out.
2. SelectKBest is fitted using only the other 131 participants and selects 50
   eGeMAPS statistical features.
3. Those 50 audio features are joined with all 10 transcript features.
4. StandardScaler and a linear SVM (`C=0.01`) are fitted on the training
   participants and evaluated on the held-out participant.

The feature count and model setting are carried forward from the already
selected individual systems; the fusion LOSO predictions are not used for
parameter tuning.

Run the text and audio systems first, then run:

```bash
python "fusion/text-audio early fusion.py"
```

Participant-level predictions are written under `data/processed/fusion/` and
remain git-ignored. Aggregate evaluation and comparison tables are written
under `results/`.

## Result

| Accuracy | Depressed precision | Depressed recall | Depressed F1 | Non-depressed F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 64.39% | 0.656 | 0.606 | 0.630 | 0.657 | 0.643 |

Confusion matrix: `[[45, 21], [26, 40]]`.

Compared with eGeMAPS alone, fusion slightly improved accuracy and macro F1
and reduced false positives, but its depressed-class recall and F1 were lower.
