import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed = os.path.join(repo, "data", "processed")

split_file = os.path.join(processed, "edaic_nlp_balanced_loso_results.csv")
text_file = os.path.join(processed, "edaic_nlp_features.csv")
audio_file = os.path.join(
    processed,
    "audio",
    "high_level_statistics",
    "edaic_egemaps_high_stats_features.csv",
)
output_file = os.path.join(
    processed,
    "fusion",
    "text_audio_fusion_evaluation.csv",
)


text_features = [
    "avg_sentiment",
    "speech_speed",
    "avg_unique_frequency",
    "avg_sw_frequency",
    "avg_characters",
    "avg_nouns",
    "avg_verbs",
    "adj_freq",
    "avg_adv",
    "fp_avg",
]

AUDIO_K = 50
SVM_C = 0.01


def load_data():
    split = pd.read_csv(split_file)[["Participant_ID", "actual_label"]]
    text = pd.read_csv(text_file)
    audio = pd.read_csv(audio_file)

    audio_features = [
        column
        for column in audio.columns
        if column.startswith("egemaps_")
        and column != "egemaps_speech_frame_count"
    ]

    data = split.merge(
        text[["Participant_ID"] + text_features],
        on="Participant_ID",
        validate="one_to_one",
    )
    data = data.merge(
        audio[["Participant_ID"] + audio_features],
        on="Participant_ID",
        validate="one_to_one",
    )

    counts = data["actual_label"].value_counts().to_dict()
    if len(data) != 132 or counts.get(0) != 66 or counts.get(1) != 66:
        raise ValueError("Expected 132 participants with a balanced 66/66 split.")
    if data[text_features + audio_features].isna().any().any():
        raise ValueError("Some text or audio features are missing.")

    return (
        data[text_features].to_numpy(dtype=float),
        data[audio_features].to_numpy(dtype=float),
        data["actual_label"].to_numpy(dtype=int),
    )


def run_fusion():
    text, audio, labels = load_data()
    predictions = []

    # Leave one participant out for testing in every fold.
    for test_index in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_index] = False

        audio_selector = Pipeline(
            [
                ("remove_constant", VarianceThreshold()),
                ("select_best", SelectKBest(f_classif, k=AUDIO_K)),
            ]
        )
        audio_train = audio_selector.fit_transform(
            audio[train_mask],
            labels[train_mask],
        )
        audio_test = audio_selector.transform(audio[test_index].reshape(1, -1))

        # Early fusion: combine 10 text and 50 selected audio features.
        fusion_train = np.column_stack([text[train_mask], audio_train])
        fusion_test = np.column_stack(
            [text[test_index].reshape(1, -1), audio_test]
        )

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("svm", SVC(kernel="linear", C=SVM_C)),
            ]
        )
        model.fit(fusion_train, labels[train_mask])
        predictions.append(int(model.predict(fusion_test)[0]))

    predictions = np.array(predictions)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    results = {
        "Modality": "Text + eGeMAPS audio",
        "Model": "Linear SVM",
        "Accuracy": accuracy_score(labels, predictions),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "F1-score": f1_score(labels, predictions, zero_division=0),
        "Non-depressed_F1": f1_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "Macro_F1": f1_score(labels, predictions, average="macro"),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame([results]).to_csv(output_file, index=False)

    print("\nText + audio early fusion")
    print("Accuracy:", round(results["Accuracy"] * 100, 2), "%")
    print("Depressed recall:", round(results["Recall"], 3))
    print("Depressed precision:", round(results["Precision"], 3))
    print("Depressed F1:", round(results["F1-score"], 3))
    print("Non-depressed F1:", round(results["Non-depressed_F1"], 3))
    print("Macro F1:", round(results["Macro_F1"], 3))
    print("Confusion matrix:", matrix.tolist())


if __name__ == "__main__":
    run_fusion()
