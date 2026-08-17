import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


repository_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(repository_folder, "data")
processed_folder = os.path.join(data_folder, "processed")
output_folder = os.path.join(processed_folder, "fusion")

balanced_split_file = os.environ.get(
    "EDAIC_BALANCED_SPLIT",
    os.path.join(data_folder, "EDAIC_balanced_66_66_split.csv"),
)
text_features_file = os.path.join(processed_folder, "edaic_nlp_features.csv")
video_119_file = os.path.join(
    processed_folder, "au_pooled_features_all_participants.csv"
)
video_217_file = os.environ.get(
    "EDAIC_VIDEO_217",
    os.path.join(processed_folder, "au_pose_gaze_features_combined_275.csv"),
)

model_choice = os.environ.get("EDAIC_FUSION_MODEL", "logreg").lower()
selected_video_count = int(os.environ.get("EDAIC_FUSION_VIDEO_K", 50))
svm_regularisation = float(os.environ.get("EDAIC_FUSION_C", 0.01))
tuning_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
random_seed = 42

# The participant set is exactly balanced, so re-weighting the classes buys
# nothing -- and under leave-one-out it actively leaks (see run_noise_check).
class_weight = os.environ.get("EDAIC_FUSION_CLASS_WEIGHT") or None

comparison_file = os.path.join(
    output_folder, "text_video_loso_" + model_choice + "_comparison.csv"
)
prediction_file = os.path.join(
    output_folder, "text_video_loso_" + model_choice + "_predictions.csv"
)
plot_file = os.path.join(
    output_folder, "text_video_loso_" + model_choice + "_comparison.png"
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

# Columns that describe the recording or the participant rather than their
# behaviour, so they are never model inputs.
video_meta_columns = [
    "Participant_ID",
    "Gender",
    "depressed_label",
    "PHQ_Score",
    "n_frames",
    "duration_s",
]

model_labels = {
    "logreg": "Logistic Regression",
    "svm": "Linear SVM (C=" + str(svm_regularisation) + ")",
    "svm-tuned": "Linear SVM (C tuned in-fold)",
}


def load_data():
    """The 132 balanced participants with their text and two video feature blocks."""
    balanced = pd.read_csv(balanced_split_file)[["Participant_ID", "depressed_label"]]
    text = pd.read_csv(text_features_file)

    data = balanced.merge(
        text[["Participant_ID", "depressed_label"] + text_feature_names],
        on="Participant_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_text"),
    )
    if len(data) != 132 or data.isna().any().any():
        raise ValueError("Fusion data must contain all 132 balanced participants")
    if not np.array_equal(data["depressed_label"], data["depressed_label_text"]):
        raise ValueError("Text labels do not match the balanced participant list")

    participant_ids = data["Participant_ID"].to_numpy(dtype=int)
    matrices = {"text": data[text_feature_names].to_numpy(dtype=float)}
    feature_names = {}

    # The two video tables share AU column names, so each block is aligned to the
    # participant order on its own rather than merged into one wide frame.
    for name, path in (("video119", video_119_file), ("video217", video_217_file)):
        table = pd.read_csv(path)
        table["Participant_ID"] = table["Participant_ID"].astype(int)
        columns = [c for c in table.columns if c not in video_meta_columns]
        aligned = table.set_index("Participant_ID").reindex(participant_ids)[columns]
        if aligned.isna().any().any():
            raise ValueError(name + " is missing some of the 132 balanced participants")
        matrices[name] = aligned.to_numpy(dtype=float)
        feature_names[name] = columns

    return (
        participant_ids,
        matrices,
        data["depressed_label"].to_numpy(dtype=int),
        feature_names,
    )


def make_model():
    """The selected classifier, with its scaler fitted inside the fold."""
    if model_choice == "logreg":
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight=class_weight,
            max_iter=5000,
            random_state=random_seed,
        )
    elif model_choice in ("svm", "svm-tuned"):
        estimator = SVC(
            kernel="linear",
            C=svm_regularisation,
            class_weight=class_weight,
            gamma="scale",
            random_state=random_seed,
        )
    else:
        raise ValueError("EDAIC_FUSION_MODEL must be logreg, svm or svm-tuned")

    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])

    if model_choice == "svm-tuned":
        # The search sees only the rows it is handed, i.e. the training fold.
        return GridSearchCV(
            pipeline,
            {"model__C": tuning_grid},
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
    return pipeline


def make_video_selector(k):
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def fit_and_score(x_train, y_train, x_test):
    """Fit one fold and return (hard predictions, comparable [0,1] scores).

    Logistic regression gives calibrated probabilities directly. The SVM has no
    probabilities, so its decision values are divided by the spread of the
    *training* decision values and squashed -- a monotone rescaling, so it leaves
    ranking (and therefore AUC) untouched, keeps the SVM's own 0 boundary at 0.5,
    and puts the two modalities on a comparable scale for late fusion.
    """
    model = make_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).astype(int)

    if hasattr(model, "predict_proba") and model_choice == "logreg":
        return predictions, model.predict_proba(x_test)[:, 1]

    train_decisions = model.decision_function(x_train)
    spread = float(np.std(train_decisions))
    if spread <= 0:
        spread = 1.0
    scores = 1.0 / (1.0 + np.exp(-model.decision_function(x_test) / spread))
    return predictions, scores


# --- fold builders: (train_mask, test_indices) -> (x_train, x_test) -------------
# Anything that learns is fitted on the training rows only.


def plain_fold(values):
    def build(train_mask, test_indices):
        return values[train_mask], values[test_indices]

    return build


def early_fusion_fold(text_values, video_values):
    def build(train_mask, test_indices):
        x_train = np.column_stack([text_values[train_mask], video_values[train_mask]])
        x_test = np.column_stack(
            [text_values[test_indices], video_values[test_indices]]
        )
        return x_train, x_test

    return build


def selected_early_fusion_fold(text_values, video_values, y_values, k):
    def build(train_mask, test_indices):
        selector = make_video_selector(k)
        video_train = selector.fit_transform(
            video_values[train_mask], y_values[train_mask]
        )
        video_test = selector.transform(video_values[test_indices])
        x_train = np.column_stack([text_values[train_mask], video_train])
        x_test = np.column_stack([text_values[test_indices], video_test])
        return x_train, x_test

    return build


def run_loso(y_values, build_fold):
    """Leave-one-subject-out over the balanced set."""
    predictions = np.zeros(len(y_values), dtype=int)
    scores = np.zeros(len(y_values), dtype=float)

    for test_index in range(len(y_values)):
        train_mask = np.ones(len(y_values), dtype=bool)
        train_mask[test_index] = False
        test_indices = np.array([test_index])
        x_train, x_test = build_fold(train_mask, test_indices)
        fold_predictions, fold_scores = fit_and_score(
            x_train, y_values[train_mask], x_test
        )
        predictions[test_index] = fold_predictions[0]
        scores[test_index] = fold_scores[0]

    return predictions, scores


def run_loso_late_fusion(y_values, text_values, video_values):
    """One model per modality per fold, then average their scores."""
    scores = np.zeros(len(y_values), dtype=float)

    for test_index in range(len(y_values)):
        train_mask = np.ones(len(y_values), dtype=bool)
        train_mask[test_index] = False
        test_indices = np.array([test_index])
        modality_scores = [
            fit_and_score(block[train_mask], y_values[train_mask], block[test_indices])[
                1
            ][0]
            for block in (text_values, video_values)
        ]
        scores[test_index] = float(np.mean(modality_scores))

    return (scores >= 0.5).astype(int), scores


def run_stratified(y_values, build_fold, splits=10, repeats=10):
    """Repeated stratified k-fold: the headline protocol.

    Every training fold keeps the 50/50 class balance, so -- unlike leave-one-out
    on an exactly balanced set -- the fold composition carries no information
    about the held-out labels. Metrics are computed per repeat over the full set
    of out-of-fold predictions, then averaged across repeats.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeats, random_state=random_seed
    )
    per_repeat = []
    predictions = np.zeros(len(y_values), dtype=int)
    scores = np.zeros(len(y_values), dtype=float)

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(y_values)), y_values)
    ):
        train_mask = np.zeros(len(y_values), dtype=bool)
        train_mask[train_index] = True
        x_train, x_test = build_fold(train_mask, test_index)
        fold_predictions, fold_scores = fit_and_score(
            x_train, y_values[train_mask], x_test
        )
        predictions[test_index] = fold_predictions
        scores[test_index] = fold_scores
        if (fold_number + 1) % splits == 0:
            per_repeat.append(calculate_scores(y_values, predictions, scores))

    averaged = {}
    for metric in per_repeat[0]:
        values = [repeat[metric] for repeat in per_repeat]
        averaged[metric] = float(np.mean(values))
        if metric in ("Accuracy", "Macro_F1", "AUC"):
            averaged[metric + "_sd"] = float(np.std(values))
    return averaged


def run_stratified_late_fusion(
    y_values, text_values, video_values, splits=10, repeats=10
):
    splitter = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeats, random_state=random_seed
    )
    per_repeat = []
    scores = np.zeros(len(y_values), dtype=float)

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(y_values)), y_values)
    ):
        train_mask = np.zeros(len(y_values), dtype=bool)
        train_mask[train_index] = True
        modality_scores = [
            fit_and_score(block[train_mask], y_values[train_mask], block[test_index])[1]
            for block in (text_values, video_values)
        ]
        scores[test_index] = np.mean(modality_scores, axis=0)
        if (fold_number + 1) % splits == 0:
            per_repeat.append(
                calculate_scores(y_values, (scores >= 0.5).astype(int), scores)
            )

    averaged = {}
    for metric in per_repeat[0]:
        values = [repeat[metric] for repeat in per_repeat]
        averaged[metric] = float(np.mean(values))
        if metric in ("Accuracy", "Macro_F1", "AUC"):
            averaged[metric + "_sd"] = float(np.std(values))
    return averaged


def run_noise_check(y_values, width=10, draws=10):
    """What each protocol scores on features that contain no signal at all.

    Anything other than ~0.5 is the protocol reading the labels off the fold
    composition rather than the features, so it is the floor the real numbers
    have to be judged against. Averaged over several draws, because a single
    random matrix can correlate with the labels by luck.
    """
    generator = np.random.RandomState(random_seed)
    loso_accuracies = []
    stratified_accuracies = []
    for _ in range(draws):
        noise = generator.normal(size=(len(y_values), width))
        loso_predictions, _ = run_loso(y_values, plain_fold(noise))
        loso_accuracies.append(accuracy_score(y_values, loso_predictions))
        stratified_accuracies.append(
            run_stratified(y_values, plain_fold(noise), repeats=3)["Accuracy"]
        )
    return float(np.mean(loso_accuracies)), float(np.mean(stratified_accuracies))


def slugify(name):
    """Whole system name to a column-safe key, so no two systems collide."""
    keep = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return "_".join(part for part in keep.split("_") if part)


def calculate_scores(y_values, predictions, scores):
    matrix = confusion_matrix(y_values, predictions, labels=[0, 1])
    return {
        "Accuracy": accuracy_score(y_values, predictions),
        "Recall": recall_score(y_values, predictions, pos_label=1, zero_division=0),
        "Precision": precision_score(
            y_values, predictions, pos_label=1, zero_division=0
        ),
        "F1-score": f1_score(y_values, predictions, pos_label=1, zero_division=0),
        "Macro_Recall": recall_score(
            y_values, predictions, average="macro", zero_division=0
        ),
        "Macro_Precision": precision_score(
            y_values, predictions, average="macro", zero_division=0
        ),
        "Non-depressed_F1": f1_score(
            y_values, predictions, pos_label=0, zero_division=0
        ),
        "Macro_F1": f1_score(y_values, predictions, average="macro", zero_division=0),
        "AUC": roc_auc_score(y_values, scores),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }


def save_plot(comparison):
    labels = comparison["System"].tolist()
    positions = np.arange(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.barh(positions + width / 2, comparison["CV_Macro_F1"], width, label="Macro F1")
    axis.barh(positions - width / 2, comparison["CV_AUC"], width, label="ROC-AUC")
    axis.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title(
        "Text + video fusion, balanced 66/66, repeated stratified 10-fold\n"
        + model_labels[model_choice]
    )
    axis.legend(loc="lower right")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)


def main():
    participant_ids, matrices, y_values, feature_names = load_data()
    os.makedirs(output_folder, exist_ok=True)

    text_values = matrices["text"]
    print("Participants: 132 (66 non-depressed, 66 depressed)", flush=True)
    print("LOSO folds:", len(y_values), flush=True)
    print(
        "Features:",
        len(text_feature_names),
        "transcript |",
        len(feature_names["video119"]),
        "Action Unit |",
        len(feature_names["video217"]),
        "Action Unit + pose + gaze",
        flush=True,
    )
    print("Model:", model_labels[model_choice], flush=True)
    print("class_weight:", class_weight, flush=True)

    loso_noise, stratified_noise = run_noise_check(y_values)
    print(
        "\nNo-signal floor (10 pure-noise features, true value 0.500):"
        "\n  leave-one-out accuracy       %.3f" % loso_noise
        + "\n  stratified 10-fold accuracy  %.3f" % stratified_noise,
        flush=True,
    )
    if abs(loso_noise - 0.5) > 0.05:
        print(
            "  WARNING: leave-one-out is reading labels off the 65/66 fold"
            "\n  composition. Treat its numbers as biased; stratified is the"
            "\n  protocol to report.",
            flush=True,
        )
    print("", flush=True)

    systems = [
        ("Text only", len(text_feature_names), plain_fold(text_values)),
        (
            "Video only (119 AU)",
            len(feature_names["video119"]),
            plain_fold(matrices["video119"]),
        ),
        (
            "Video only (217 AU+pose+gaze)",
            len(feature_names["video217"]),
            plain_fold(matrices["video217"]),
        ),
        (
            "Early fusion: text + 119 AU",
            len(text_feature_names) + len(feature_names["video119"]),
            early_fusion_fold(text_values, matrices["video119"]),
        ),
        (
            "Early fusion: text + 217 AU+pose+gaze",
            len(text_feature_names) + len(feature_names["video217"]),
            early_fusion_fold(text_values, matrices["video217"]),
        ),
        (
            "Early fusion: text + top-" + str(selected_video_count) + " of 119",
            len(text_feature_names) + selected_video_count,
            selected_early_fusion_fold(
                text_values, matrices["video119"], y_values, selected_video_count
            ),
        ),
        (
            "Early fusion: text + top-" + str(selected_video_count) + " of 217",
            len(text_feature_names) + selected_video_count,
            selected_early_fusion_fold(
                text_values, matrices["video217"], y_values, selected_video_count
            ),
        ),
    ]

    rows = []
    predictions_table = {"Participant_ID": participant_ids, "actual_label": y_values}

    def row_for(name, feature_count, loso_scores, cv_scores):
        row = {
            "System": name,
            "Model": model_labels[model_choice],
            "Features": feature_count,
        }
        row.update({"CV_" + key: value for key, value in cv_scores.items()})
        row.update({"LOSO_" + key: value for key, value in loso_scores.items()})
        return row

    for name, feature_count, build_fold in systems:
        predictions, scores = run_loso(y_values, build_fold)
        rows.append(
            row_for(
                name,
                feature_count,
                calculate_scores(y_values, predictions, scores),
                run_stratified(y_values, build_fold),
            )
        )
        key = slugify(name)
        predictions_table[key + "_loso_predicted"] = predictions
        predictions_table[key + "_loso_score"] = scores
        print("done:", name, flush=True)

    for label, video_block in (
        ("Late fusion: text + 119 AU (mean score)", matrices["video119"]),
        ("Late fusion: text + 217 AU+pose+gaze (mean score)", matrices["video217"]),
    ):
        predictions, scores = run_loso_late_fusion(y_values, text_values, video_block)
        rows.append(
            row_for(
                label,
                len(text_feature_names) + video_block.shape[1],
                calculate_scores(y_values, predictions, scores),
                run_stratified_late_fusion(y_values, text_values, video_block),
            )
        )
        key = slugify(label)
        predictions_table[key + "_loso_predicted"] = predictions
        predictions_table[key + "_loso_score"] = scores
        print("done:", label, flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(comparison_file, index=False)
    pd.DataFrame(predictions_table).to_csv(prediction_file, index=False)
    save_plot(comparison)

    print(
        "\nHEADLINE: repeated stratified 10-fold (10 repeats) --",
        model_labels[model_choice],
        "\n",
    )
    print(
        comparison[
            [
                "System",
                "Features",
                "CV_Accuracy",
                "CV_Accuracy_sd",
                "CV_Macro_Precision",
                "CV_Macro_Recall",
                "CV_Macro_F1",
                "CV_AUC",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )

    print(
        "\nSECONDARY: leave-one-subject-out (biased on an exactly balanced set,"
        "\nsee the no-signal floor above) --",
        model_labels[model_choice],
        "\n",
    )
    print(
        comparison[
            [
                "System",
                "Features",
                "LOSO_Accuracy",
                "LOSO_Macro_Precision",
                "LOSO_Macro_Recall",
                "LOSO_Macro_F1",
                "LOSO_AUC",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )
    print("\nWrote results to", output_folder, flush=True)


if __name__ == "__main__":
    main()
