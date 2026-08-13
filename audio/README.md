# Audio pipeline

The final speech system keeps participant speech frames from the supplied
openSMILE eGeMAPS files and summarizes each acoustic feature using mean,
standard deviation, median, percentiles, IQR, MAD, skewness, and kurtosis.

Inside every leave-one-subject-out training fold, ANOVA SelectKBest chooses 50
features. A standardized linear SVM (`C=0.1`) then predicts the held-out
participant. It uses the exact balanced 132-person list created by the text
system.

Run the text system first, then run from the repository root:

```bash
python audio/edaic_audio_system.py
```

Final result: 63.64% accuracy, 0.642 depressed-class F1, and 0.636 macro F1.
