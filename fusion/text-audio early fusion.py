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


repository_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_folder = os.path.join(repository_folder, "data", "processed")
audio_statistics_file = os.path.join(
    processed_folder,
    "audio",
    "high_level_statistics",
    "edaic_egemaps_high_stats_features.csv",
)
text_features_file = os.path.join(
    processed_folder,
    "edaic_nlp_features.csv",
)
balanced_file = os.path.join(
    processed_folder,
    "edaic_nlp_balanced_loso_results.csv",
)
text_evaluation_file = os.path.join(
    processed_folder,
    "edaic_nlp_balanced_loso_model_evaluation.csv",
)
audio_evaluation_file = os.path.join(
    processed_folder,
    "audio",
    "high_level_statistics",
    "edaic_audio_balanced_loso_evaluation.csv",
)
fusion_output_folder = os.path.join(processed_folder, "fusion")
prediction_file = os.path.join(
    fusion_output_folder,
    "text_audio_fusion_predictions.csv",
)
selected_features_file = os.path.join(
    fusion_output_folder,
    "text_audio_fusion_selected_audio_features.csv",
)
results_folder = os.path.join(repository_folder, "results")
evaluation_file = os.path.join(
    results_folder,
    "text_audio_fusion_evaluation.csv",
)
comparison_file = os.path.join(
    results_folder,
    "text_audio_fusion_comparison.csv",
)

text_feature_names = [
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


def load_data():
    balanced = pd.read_csv(balanced_file)[
        ["Participant_ID", "actual_label"]
    ]
    text = pd.read_csv(text_features_file)
    audio = pd.read_csv(audio_statistics_file)
    data = balanced.merge(
        text,
        on="Participant_ID",
        how="left",
        validate="one_to_one",
    ).merge(
        audio,
        on="Participant_ID",
        how="left",
        validate="one_to_one",
        suffixes=("_text", "_audio"),
    )

    if len(data) != 132 or data.isna().any().any():
        raise ValueError("Fusion data must contain all 132 balanced participants")

    if not np.array_equal(data["actual_label"], data["label_text"]):
        raise ValueError("Text labels do not match the balanced participant list")

    if not np.array_equal(data["actual_label"], data["label_audio"]):
        raise ValueError("Audio labels do not match the balanced participant list")

    audio_feature_names = [
        column
        for column in audio.columns
        if column.startswith("egemaps_")
        and not column.endswith("speech_frame_count")
    ]
    return (
        data["Participant_ID"].to_numpy(dtype=int),
        data[text_feature_names].to_numpy(dtype=float),
        data[audio_feature_names].to_numpy(dtype=float),
        data["actual_label"].to_numpy(dtype=int),
        audio_feature_names,
    )


def make_audio_selector():
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=50)),
        ]
    )


def make_fusion_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(kernel="linear", C=0.01, gamma="scale")),
        ]
    )


def selected_audio_features(selector, audio_feature_names):
    names_after_variance = np.asarray(audio_feature_names)[
        selector.named_steps["variance"].get_support()
    ]
    select_step = selector.named_steps["select"]
    selected_names = names_after_variance[select_step.get_support()]
    selected_scores = select_step.scores_[select_step.get_support()]
    return [
        (str(name), float(score) if np.isfinite(score) else 0.0)
        for name, score in zip(selected_names, selected_scores)
    ]


def calculate_scores(y_values, predictions):
    matrix = confusion_matrix(y_values, predictions, labels=[0, 1])
    return {
        "Accuracy": accuracy_score(y_values, predictions),
        "Recall": recall_score(
            y_values,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_values,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_values,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "Non-depressed_F1": f1_score(
            y_values,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "Macro_F1": f1_score(
            y_values,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }


def save_comparison(fusion_scores):
    rows = []

    if os.path.exists(text_evaluation_file):
        text = pd.read_csv(text_evaluation_file).iloc[0]
        rows.append(
            {
                "System": "Transcript",
                "Model": "Linear SVM",
                "Accuracy": text["Accuracy"],
                "Recall": text["Recall"],
                "Precision": text["Precision"],
                "F1-score": text["F1-score"],
                "Non-depressed_F1": text["Non-depressed_F1"],
                "Macro_F1": text["Macro_F1"],
            }
        )

    if os.path.exists(audio_evaluation_file):
        audio = pd.read_csv(audio_evaluation_file).iloc[0]
        rows.append(
            {
                "System": "eGeMAPS audio",
                "Model": "Linear SVM",
                "Accuracy": audio["Accuracy"],
                "Recall": audio["Recall"],
                "Precision": audio["Precision"],
                "F1-score": audio["F1-score"],
                "Non-depressed_F1": audio["Non-depressed_F1"],
                "Macro_F1": audio["Macro_F1"],
            }
        )

    rows.append(
        {
            "System": "Text + eGeMAPS feature fusion",
            "Model": "Linear SVM",
            **{
                name: fusion_scores[name]
                for name in [
                    "Accuracy",
                    "Recall",
                    "Precision",
                    "F1-score",
                    "Non-depressed_F1",
                    "Macro_F1",
                ]
            },
        }
    )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(comparison_file, index=False)
    print("\nSystem comparison", flush=True)
    print(comparison.to_string(index=False), flush=True)


def run_fusion():
    (
        participant_ids,
        text_values,
        audio_values,
        y_values,
        audio_feature_names,
    ) = load_data()
    predictions = []
    decision_scores = []
    selected_counts = {}
    selected_score_totals = {}

    for test_index in range(len(y_values)):
        training_mask = np.ones(len(y_values), dtype=bool)
        training_mask[test_index] = False
        selector = make_audio_selector()
        selected_audio_train = selector.fit_transform(
            audio_values[training_mask],
            y_values[training_mask],
        )
        selected_audio_test = selector.transform(
            audio_values[test_index].reshape(1, -1)
        )
        x_train = np.column_stack(
            [text_values[training_mask], selected_audio_train]
        )
        x_test = np.column_stack(
            [text_values[test_index].reshape(1, -1), selected_audio_test]
        )
        model = make_fusion_model()
        model.fit(x_train, y_values[training_mask])
        predictions.append(int(model.predict(x_test)[0]))
        decision_scores.append(float(model.decision_function(x_test)[0]))

        for name, score in selected_audio_features(selector, audio_feature_names):
            selected_counts[name] = selected_counts.get(name, 0) + 1
            selected_score_totals[name] = (
                selected_score_totals.get(name, 0.0) + score
            )

        if (test_index + 1) % 22 == 0:
            print(
                "Completed",
                test_index + 1,
                "of",
                len(y_values),
                "folds",
                flush=True,
            )

    predictions = np.asarray(predictions, dtype=int)
    scores = calculate_scores(y_values, predictions)
    os.makedirs(fusion_output_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    pd.DataFrame(
        {
            "Participant_ID": participant_ids,
            "actual_label": y_values,
            "predicted_label": predictions,
            "decision_score": decision_scores,
        }
    ).to_csv(prediction_file, index=False)
    pd.DataFrame(
        [
            {
                "feature": name,
                "folds_selected": count,
                "selection_frequency": count / len(y_values),
                "mean_training_f_score_when_selected": (
                    selected_score_totals[name] / count
                ),
            }
            for name, count in selected_counts.items()
        ]
    ).sort_values(
        ["folds_selected", "mean_training_f_score_when_selected"],
        ascending=False,
    ).to_csv(selected_features_file, index=False)
    evaluation = pd.DataFrame(
        [
            {
                "Method": "Feature-level fusion",
                "Modalities": "10 transcript + 50 selected eGeMAPS",
                "Model": "Linear SVM",
                "Participants": len(y_values),
                "C": 0.01,
                **scores,
            }
        ]
    )
    evaluation.to_csv(evaluation_file, index=False)

    print("\nText + eGeMAPS feature fusion", flush=True)
    print("Participants: 132 (66 non-depressed, 66 depressed)", flush=True)
    print("LOSO folds:", len(y_values), flush=True)
    print("Accuracy:", round(scores["Accuracy"] * 100, 2), "%", flush=True)
    print("Depressed precision:", round(scores["Precision"], 3), flush=True)
    print("Depressed recall:", round(scores["Recall"], 3), flush=True)
    print("Depressed F1:", round(scores["F1-score"], 3), flush=True)
    print("Non-depressed F1:", round(scores["Non-depressed_F1"], 3), flush=True)
    print("Macro F1:", round(scores["Macro_F1"], 3), flush=True)
    print(
        "Confusion matrix:",
        [[scores["TN"], scores["FP"]], [scores["FN"], scores["TP"]]],
        flush=True,
    )
    save_comparison(scores)


if __name__ == "__main__":
    run_fusion()
