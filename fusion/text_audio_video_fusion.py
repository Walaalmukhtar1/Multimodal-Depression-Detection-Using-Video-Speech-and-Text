import glob
import io
import os
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


repository_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(repository_folder, "data")
processed_folder = os.path.join(data_folder, "processed")
audio_folder = os.path.join(processed_folder, "audio")
output_folder = os.path.join(processed_folder, "fusion")

balanced_split_file = os.path.join(data_folder, "EDAIC_balanced_66_66_split.csv")
text_features_file = os.path.join(processed_folder, "edaic_nlp_features.csv")
video_119_file = os.path.join(
    processed_folder, "au_pooled_features_all_participants.csv"
)
video_217_file = os.environ.get(
    "EDAIC_VIDEO_217",
    os.path.join(
        r"D:\Downloads\lightweight-depression-detection-3d-landmarks-main",
        "analysis_handover",
        "au_pose_gaze_features_combined_275.csv",
    ),
)
edaic_raw_folder = os.environ.get(
    "EDAIC_RAW_DIR",
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\e-daic-woz",
)
audio_cache_file = os.path.join(audio_folder, "edaic_egemaps_pooled_features.csv")
audio_features_file = os.environ.get("EDAIC_AUDIO_FEATURES", audio_cache_file)

model_choice = os.environ.get("EDAIC_FUSION_MODEL", "logreg").lower()
block_feature_count = int(os.environ.get("EDAIC_FUSION_K", 50))
random_seed = 42

comparison_file = os.path.join(
    output_folder, "text_audio_video_" + model_choice + "_comparison.csv"
)
plot_file = os.path.join(
    output_folder, "text_audio_video_" + model_choice + "_comparison.png"
)

model_labels = {
    "logreg": "Logistic Regression (C=1.0)",
    "svm": "Linear SVM (C=0.01)",
}

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

video_meta_columns = [
    "Participant_ID",
    "Gender",
    "depressed_label",
    "PHQ_Score",
    "n_frames",
    "duration_s",
]

statistic_names = ["mean", "std", "var", "min", "max", "skew", "kurt"]


def pool_statistics(values):
    """The video pipeline's 7 statistics, column-wise."""
    return np.concatenate(
        [
            np.nanmean(values, axis=0),
            np.nanstd(values, axis=0),
            np.nanvar(values, axis=0),
            np.nanmin(values, axis=0),
            np.nanmax(values, axis=0),
            skew(values, axis=0, nan_policy="omit"),
            kurtosis(values, axis=0, nan_policy="omit"),
        ]
    )


def pooled_feature_names(descriptors):
    return [
        descriptor + "_" + statistic
        for statistic in statistic_names
        for descriptor in descriptors
    ]


def read_egemaps(archive_path):
    """Pull the eGeMAPS frame table out of one E-DAIC archive."""
    with tarfile.open(archive_path) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("OpenSMILE2.3.0_egemaps.csv")
        ]
        if not members:
            return None
        raw = archive.extractfile(members[0]).read()

    frames = pd.read_csv(io.BytesIO(raw), sep=";")
    # 'name' is the recording label and 'frameTime' is the timestamp; neither
    # describes the speech itself.
    return frames.drop(columns=[c for c in ("name", "frameTime") if c in frames])


def build_audio_features(wanted_ids):
    """Pool eGeMAPS frames per participant. Cached after the first run."""
    if os.path.exists(audio_features_file):
        return pd.read_csv(audio_features_file)

    print(
        "Pooling eGeMAPS frames from the E-DAIC archives"
        " (first run only, ~20 minutes)...",
        flush=True,
    )
    os.makedirs(audio_folder, exist_ok=True)
    rows = []
    descriptors = None

    archives = sorted(glob.glob(os.path.join(edaic_raw_folder, "*_P.tar.gz")))
    archives = [
        path
        for path in archives
        if int(os.path.basename(path).split("_")[0]) in wanted_ids
    ]

    for position, path in enumerate(archives, start=1):
        participant_id = int(os.path.basename(path).split("_")[0])
        frames = read_egemaps(path)
        if frames is None or len(frames) < 100:
            print("  skipping", participant_id, "- no usable eGeMAPS frames")
            continue
        if descriptors is None:
            descriptors = list(frames.columns)

        rows.append(
            dict(
                zip(
                    ["Participant_ID"] + pooled_feature_names(descriptors),
                    np.concatenate(
                        [[participant_id], pool_statistics(frames.to_numpy(float))]
                    ),
                )
            )
        )
        if position % 10 == 0:
            print("  pooled", position, "of", len(archives), flush=True)

    table = pd.DataFrame(rows)
    table["Participant_ID"] = table["Participant_ID"].astype(int)
    table.to_csv(audio_features_file, index=False)
    print("Wrote", audio_features_file, table.shape, flush=True)
    return table


def align(table, participant_ids, drop_columns):
    columns = [c for c in table.columns if c not in drop_columns]
    aligned = table.set_index("Participant_ID").reindex(participant_ids)[columns]
    return aligned.to_numpy(dtype=float), columns


def load_data():
    balanced = pd.read_csv(balanced_split_file)[["Participant_ID", "depressed_label"]]
    text = pd.read_csv(text_features_file)
    audio = build_audio_features(set(balanced["Participant_ID"]))
    audio["Participant_ID"] = audio["Participant_ID"].astype(int)

    # Restrict to participants that have all three modalities.
    usable = balanced[balanced["Participant_ID"].isin(set(audio["Participant_ID"]))]
    usable = usable.merge(
        text[["Participant_ID"] + text_feature_names],
        on="Participant_ID",
        how="inner",
        validate="one_to_one",
    ).sort_values("Participant_ID")

    participant_ids = usable["Participant_ID"].to_numpy(dtype=int)
    y_values = usable["depressed_label"].to_numpy(dtype=int)

    matrices = {"text": usable[text_feature_names].to_numpy(dtype=float)}
    matrices["audio"], audio_names = align(
        audio, participant_ids, ["Participant_ID"]
    )
    video = pd.read_csv(video_119_file)
    video["Participant_ID"] = video["Participant_ID"].astype(int)
    matrices["video"], video_names = align(video, participant_ids, video_meta_columns)

    for name, block in matrices.items():
        if not np.isfinite(block).all():
            column_ok = np.isfinite(block).all(axis=0)
            print(
                "  dropping",
                int((~column_ok).sum()),
                "non-finite columns from",
                name,
            )
            matrices[name] = block[:, column_ok]

    return participant_ids, matrices, y_values


def make_model():
    if model_choice == "logreg":
        estimator = LogisticRegression(
            C=1.0, penalty="l2", max_iter=5000, random_state=random_seed
        )
    elif model_choice == "svm":
        estimator = SVC(
            kernel="linear", C=0.01, gamma="scale", random_state=random_seed
        )
    else:
        raise ValueError("EDAIC_FUSION_MODEL must be logreg or svm")
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def make_selector(k):
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def fit_and_score(x_train, y_train, x_test):
    model = make_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).astype(int)
    if model_choice == "logreg":
        return predictions, model.predict_proba(x_test)[:, 1]
    train_decisions = model.decision_function(x_train)
    spread = float(np.std(train_decisions)) or 1.0
    return predictions, 1.0 / (1.0 + np.exp(-model.decision_function(x_test) / spread))


def block_fold(blocks, y_values, select_blocks):
    """Concatenate the named blocks, selecting inside the fold where asked."""

    def build(train_mask, test_indices):
        train_parts = []
        test_parts = []
        for name, values in blocks:
            if name in select_blocks and values.shape[1] > block_feature_count:
                selector = make_selector(block_feature_count)
                train_parts.append(
                    selector.fit_transform(values[train_mask], y_values[train_mask])
                )
                test_parts.append(selector.transform(values[test_indices]))
            else:
                train_parts.append(values[train_mask])
                test_parts.append(values[test_indices])
        return np.column_stack(train_parts), np.column_stack(test_parts)

    return build


def run_stratified(y_values, build_fold, splits=10, repeats=10):
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
        predictions[test_index], scores[test_index] = fit_and_score(
            x_train, y_values[train_mask], x_test
        )
        if (fold_number + 1) % splits == 0:
            per_repeat.append(calculate_scores(y_values, predictions, scores))

    averaged = {}
    for metric in per_repeat[0]:
        values = [repeat[metric] for repeat in per_repeat]
        averaged[metric] = float(np.mean(values))
        if metric in ("Accuracy", "Macro_F1", "AUC"):
            averaged[metric + "_sd"] = float(np.std(values))
    return averaged


def run_late_fusion(y_values, blocks, splits=10, repeats=10):
    """One model per modality per fold, then average their scores."""
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
        modality_scores = []
        for name, values in blocks:
            build = block_fold([(name, values)], y_values, {name})
            x_train, x_test = build(train_mask, test_index)
            modality_scores.append(
                fit_and_score(x_train, y_values[train_mask], x_test)[1]
            )
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


def run_loso(y_values, build_fold):
    """Leave-one-subject-out.

    Reported only for comparability with the existing LOO rows. On an exactly
    balanced set every training fold is 65/66, so the fold composition alone
    points at the held-out label -- see the no-signal floor printed by main().
    """
    predictions = np.zeros(len(y_values), dtype=int)
    scores = np.zeros(len(y_values), dtype=float)

    for test_index in range(len(y_values)):
        train_mask = np.ones(len(y_values), dtype=bool)
        train_mask[test_index] = False
        x_train, x_test = build_fold(train_mask, np.array([test_index]))
        fold_predictions, fold_scores = fit_and_score(
            x_train, y_values[train_mask], x_test
        )
        predictions[test_index] = fold_predictions[0]
        scores[test_index] = fold_scores[0]

    return calculate_scores(y_values, predictions, scores)


def run_loso_late_fusion(y_values, blocks):
    scores = np.zeros(len(y_values), dtype=float)

    for test_index in range(len(y_values)):
        train_mask = np.ones(len(y_values), dtype=bool)
        train_mask[test_index] = False
        modality_scores = []
        for name, values in blocks:
            build = block_fold([(name, values)], y_values, {name})
            x_train, x_test = build(train_mask, np.array([test_index]))
            modality_scores.append(
                fit_and_score(x_train, y_values[train_mask], x_test)[1][0]
            )
        scores[test_index] = float(np.mean(modality_scores))

    return calculate_scores(y_values, (scores >= 0.5).astype(int), scores)


def run_noise_check(y_values, width=10, draws=10):
    """What each protocol scores on features carrying no signal at all."""
    generator = np.random.RandomState(random_seed)
    stratified_accuracies = []
    loso_accuracies = []
    for _ in range(draws):
        noise = generator.normal(size=(len(y_values), width))
        build = block_fold([("noise", noise)], y_values, set())
        stratified_accuracies.append(
            run_stratified(y_values, build, repeats=3)["Accuracy"]
        )
        loso_accuracies.append(run_loso(y_values, build)["Accuracy"])
    return float(np.mean(stratified_accuracies)), float(np.mean(loso_accuracies))


def calculate_scores(y_values, predictions, scores):
    matrix = confusion_matrix(y_values, predictions, labels=[0, 1])
    return {
        "Accuracy": accuracy_score(y_values, predictions),
        "Macro_Precision": precision_score(
            y_values, predictions, average="macro", zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_values, predictions, average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(y_values, predictions, average="macro", zero_division=0),
        "Depressed_F1": f1_score(y_values, predictions, pos_label=1, zero_division=0),
        "AUC": roc_auc_score(y_values, scores),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }


def save_plot(comparison, noise_floor):
    positions = np.arange(len(comparison))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.barh(positions + width / 2, comparison["Macro_F1"], width, label="Macro F1")
    axis.barh(positions - width / 2, comparison["AUC"], width, label="ROC-AUC")
    axis.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance")
    axis.axvline(
        noise_floor,
        color="firebrick",
        linestyle=":",
        linewidth=1.5,
        label="No-signal floor",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(comparison["System"], fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title(
        "Text + audio + video fusion, "
        + str(comparison.attrs.get("participants", ""))
        + " participants, repeated stratified 10-fold\n"
        + model_labels[model_choice]
    )
    axis.legend(loc="lower right")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)


def main():
    participant_ids, matrices, y_values = load_data()
    os.makedirs(output_folder, exist_ok=True)

    text = ("text", matrices["text"])
    audio = ("audio", matrices["audio"])
    video = ("video", matrices["video"])

    print("\nParticipants:", len(y_values), dict(zip(*np.unique(y_values, return_counts=True))))
    print(
        "Features:",
        matrices["text"].shape[1],
        "text |",
        matrices["audio"].shape[1],
        "audio |",
        matrices["video"].shape[1],
        "video",
    )
    print("Model:", model_labels[model_choice])
    print("Audio/video blocks reduced to", block_feature_count, "features in-fold\n")

    noise_floor, loso_noise_floor = run_noise_check(y_values)
    print("No-signal floor (pure noise, true value 0.500):")
    print("  stratified 10-fold  %.3f" % noise_floor)
    print("  leave-one-out       %.3f" % loso_noise_floor)
    if abs(loso_noise_floor - 0.5) > 0.05:
        print(
            "  WARNING: leave-one-out is reading labels off the 65/66 fold"
            " composition;\n  its rows below are biased by roughly this much."
        )
    print("", flush=True)

    systems = [
        ("Text only", [text], set()),
        ("Audio only", [audio], {"audio"}),
        ("Video only", [video], {"video"}),
        ("Text + audio", [text, audio], {"audio"}),
        ("Text + video", [text, video], {"video"}),
        ("Audio + video", [audio, video], {"audio", "video"}),
        ("Text + audio + video", [text, audio, video], {"audio", "video"}),
    ]

    rows = []
    for name, blocks, select in systems:
        build = block_fold(blocks, y_values, select)
        row = {"System": name, "Model": model_labels[model_choice]}
        row.update(run_stratified(y_values, build))
        row.update(
            {"LOSO_" + key: value for key, value in run_loso(y_values, build).items()}
        )
        rows.append(row)
        print("done:", name, flush=True)

    row = {
        "System": "Text + audio + video (late fusion)",
        "Model": model_labels[model_choice],
    }
    row.update(run_late_fusion(y_values, [text, audio, video]))
    row.update(
        {
            "LOSO_" + key: value
            for key, value in run_loso_late_fusion(
                y_values, [text, audio, video]
            ).items()
        }
    )
    rows.append(row)
    print("done: late fusion", flush=True)

    comparison = pd.DataFrame(rows)
    comparison.attrs["participants"] = len(y_values)
    comparison.to_csv(comparison_file, index=False)
    save_plot(comparison, noise_floor)

    print(
        "\nRepeated stratified 10-fold, "
        + str(len(y_values))
        + " participants --",
        model_labels[model_choice],
        "\n",
    )
    print(
        comparison[
            [
                "System",
                "Accuracy",
                "Accuracy_sd",
                "Macro_Precision",
                "Macro_Recall",
                "Macro_F1",
                "AUC",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )

    print(
        "\nSame systems under leave-one-out (biased, floor %.3f above) --"
        % loso_noise_floor,
        model_labels[model_choice],
        "\n",
    )
    print(
        comparison[
            [
                "System",
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
