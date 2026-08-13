# -*- coding: utf-8 -*-
"""Step 3 — train and evaluate the final video model.

Plain logistic regression on the 119 pooled Action Unit features:
no grid search, no feature selection, no dimensionality reduction.

Follows the official E-DAIC protocol — train on train+dev combined (219
participants), evaluate once on the held-out test set (56 participants).

Usage:
    python video/train_logreg.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RESULTS_DIR, META_COLS, RANDOM_SEED, ensure_dirs  # noqa: E402


def load_splits():
    paths = {n: PROCESSED_DIR / f"au_features_{n}.csv" for n in ("train", "dev", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Missing feature tables — run video/merge_labels.py first:\n  "
                 + "\n  ".join(missing))
    return {n: pd.read_csv(p) for n, p in paths.items()}


def main():
    ensure_dirs()
    splits = load_splits()

    # train+dev combined is the training set; test is touched exactly once
    train = pd.concat([splits["train"], splits["dev"]], ignore_index=True)
    test = splits["test"]

    feature_cols = [c for c in train.columns if c not in META_COLS]
    X_train, y_train = train[feature_cols].values, train["depressed_label"].values
    X_test, y_test = test[feature_cols].values, test["depressed_label"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Train (train+dev): {len(train)} participants, "
          f"depressed={int(y_train.sum())}")
    print(f"Test (held out):   {len(test)} participants, "
          f"depressed={int(y_test.sum())}")

    # the scaler is fit on training data only
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                             max_iter=5000, random_state=RANDOM_SEED)
    clf.fit(scaler.transform(X_train), y_train)

    y_pred = clf.predict(scaler.transform(X_test))
    y_proba = clf.predict_proba(scaler.transform(X_test))[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1])
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n=== Held-out test results ===")
    print(f"Accuracy         {acc:.3f}")
    print(f"Macro F1         {macro_f1:.3f}")
    print(f"ROC-AUC          {auc:.3f}")
    print(f"Macro precision  {macro_p:.3f}")
    print(f"Macro recall     {macro_r:.3f}")
    print("\n                 not depressed   depressed")
    print(f"Precision            {prec[0]:.3f}         {prec[1]:.3f}")
    print(f"Recall               {rec[0]:.3f}         {rec[1]:.3f}")
    print(f"F1                   {f1[0]:.3f}         {f1[1]:.3f}")
    print(f"Support              {support[0]:<13d} {support[1]}")
    print(f"\nConfusion matrix [rows=true, cols=pred]:\n{cm}")

    pd.DataFrame([{
        "accuracy": acc, "macro_f1": macro_f1, "auc": auc,
        "macro_precision": macro_p, "macro_recall": macro_r,
        "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
    }]).to_csv(RESULTS_DIR / "logreg_test_metrics.csv", index=False)

    # top standardised coefficients, as a readable summary of what drives the model
    coefs = pd.Series(clf.coef_[0], index=feature_cols).sort_values(
        key=np.abs, ascending=False)
    coefs.to_csv(RESULTS_DIR / "logreg_coefficients.csv", header=["coefficient"])
    print("\nTop 10 features by |standardised coefficient|:")
    print(coefs.head(10).to_string())

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not depressed", "depressed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not depressed", "depressed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Logistic regression — held-out test\n"
                 f"Acc={acc:.2f}  Macro-F1={macro_f1:.2f}  AUC={auc:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "logreg_confusion_matrix.png", dpi=150)
    print(f"\nSaved metrics, coefficients and confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
