# Text pipeline

The final transcript system extracts ten participant-level NLP features and
evaluates a linear SVM (`C=0.01`) with leave-one-subject-out validation.

The corpus is downsampled with random state 42 to 66 depressed and 66
non-depressed participants. This creates the shared 132-person participant
list used by the audio and future fusion systems.

Run from the repository root:

```bash
python text/edaic_nlp_system.py
```

Final result: 58.33% accuracy, 0.636 depressed-class F1, and 0.575 macro F1.
