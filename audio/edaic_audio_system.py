import io
import os
import re
import sys
import tarfile

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


project_folder = os.path.dirname(os.path.abspath(__file__))
repository_folder = os.path.dirname(project_folder)
default_dataset_folder = os.path.abspath(
    os.path.join(repository_folder, "data", "raw")
)
dataset_folder = os.environ.get(
    "EDAIC_RAW_DIR",
    os.environ.get("EDAIC_DATASET_PATH", default_dataset_folder),
)
processed_folder = os.path.join(repository_folder, "data", "processed")
transcript_cache = os.path.join(processed_folder, "transcripts")
nlp_feature_file = os.path.join(processed_folder, "edaic_nlp_features.csv")
balanced_file = os.path.join(
    processed_folder,
    "edaic_nlp_balanced_loso_results.csv",
)
audio_folder = os.path.join(processed_folder, "audio")
raw_audio_cache = os.path.join(audio_folder, "raw_cache")
statistics_folder = os.path.join(audio_folder, "high_level_statistics")
statistics_file = os.path.join(
    statistics_folder,
    "edaic_egemaps_high_stats_features.csv",
)
checkpoint_file = os.path.join(
    statistics_folder,
    "edaic_egemaps_high_stats_partial.csv",
)
evaluation_file = os.path.join(
    statistics_folder,
    "edaic_audio_balanced_loso_evaluation.csv",
)
prediction_file = os.path.join(
    statistics_folder,
    "edaic_audio_balanced_loso_predictions.csv",
)
selected_file = os.path.join(
    statistics_folder,
    "edaic_audio_balanced_loso_selected_features.csv",
)

generic_archive_ids = {
    "P Archive from USC.gz": 677,
    "P Archive.tar (8).gz": 666,
    "P Archive.tar (9).gz": 667,
    "P Archive from USC (1).gz": 698,
    "P.tar.gz": 634,
    "P Archive.tar (12).gz": 695,
    "P Archive.tar.gz": 632,
    "P Archive.tar (13).gz": 696,
    "P Archive.tar (11).gz": 691,
    "P Archive.tar (10).gz": 669,
    "P Archive.tar (14).gz": 712,
    "P Archive.tar (15).gz": 717,
    "P Archive.tar (7).gz": 659,
    "P.tar (1).gz": 697,
    "P Archive.tar (6).gz": 658,
    "P Dataset.tar.gz": 687,
    "P Archive.tar (4).gz": 656,
    "P.tar (2).gz": 699,
    "P Archive.tar (5).gz": 657,
    "P Archive.tar (1).gz": 636,
    "P Archive.tar (2).gz": 638,
    "P Archive.tar (3).gz": 637,
}


def read_participants():
    features = pd.read_csv(nlp_feature_file)
    balanced = pd.read_csv(balanced_file)[
        ["Participant_ID", "actual_label"]
    ]
    data = balanced.merge(
        features[["Participant_ID", "split", "label", "phq_score"]],
        on="Participant_ID",
        how="left",
        validate="one_to_one",
    )

    if len(data) != 132 or data.isna().any().any():
        raise ValueError("The shared balanced split must contain 132 participants")

    if not np.array_equal(data["actual_label"], data["label"]):
        raise ValueError("The transcript and audio labels do not match")

    return data[
        ["Participant_ID", "split", "label", "phq_score"]
    ].to_dict("records")


def find_transcripts():
    files = {}

    for search_folder in [dataset_folder, transcript_cache]:
        if not os.path.isdir(search_folder):
            continue

        for folder, _, names in os.walk(search_folder):
            for name in names:
                match = re.fullmatch(r"(\d+)_Transcript\.csv", name, re.I)
                if match:
                    files[int(match.group(1))] = os.path.join(folder, name)

    return files


def find_egemaps_files():
    files = {}
    pattern = r"(\d+)_OpenSMILE2\.3\.0_egemaps\.csv"

    for search_folder in [dataset_folder, raw_audio_cache]:
        if not os.path.isdir(search_folder):
            continue

        for folder, _, names in os.walk(search_folder):
            for name in names:
                match = re.fullmatch(pattern, name, re.I)
                if match:
                    files[int(match.group(1))] = os.path.join(folder, name)

    return files


def find_archives(valid_ids):
    archive_map = {}

    for search_folder in [dataset_folder, raw_audio_cache]:
        if not os.path.isdir(search_folder):
            continue

        for name in os.listdir(search_folder):
            if not name.lower().endswith((".gz", ".tgz")):
                continue

            participant_id = None
            for value in re.findall(r"(?<!\d)(\d{3})(?!\d)", name):
                if int(value) in valid_ids:
                    participant_id = int(value)
                    break

            if participant_id is None:
                participant_id = generic_archive_ids.get(name)

            if participant_id in valid_ids:
                archive_map[participant_id] = os.path.join(search_folder, name)

    return archive_map


def read_egemaps_file(path):
    return pd.read_csv(path, sep=";", low_memory=False)


def read_egemaps_archive(participant_id, archive_path):
    wanted = (
        str(participant_id) + "_OpenSMILE2.3.0_egemaps.csv"
    ).lower()

    try:
        with tarfile.open(archive_path, "r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if os.path.basename(member.name).lower() != wanted:
                    continue

                source = archive.extractfile(member)
                if source is not None:
                    return pd.read_csv(
                        io.BytesIO(source.read()),
                        sep=";",
                        low_memory=False,
                    )
    except (tarfile.TarError, OSError, pd.errors.ParserError):
        return None

    return None


def merge_intervals(transcript):
    intervals = []

    for start, end in transcript[["Start_Time", "End_Time"]].itertuples(
        index=False,
        name=None,
    ):
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue

        duration = end - start
        if np.isfinite(start) and np.isfinite(end) and 0 < duration <= 120:
            intervals.append([start, end])

    intervals.sort()
    merged = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def keep_participant_frames(frame_data, transcript):
    frame_data.columns = [str(column).strip() for column in frame_data.columns]

    if "frameTime" not in frame_data.columns:
        return pd.DataFrame()

    frame_data["frameTime"] = pd.to_numeric(
        frame_data["frameTime"],
        errors="coerce",
    )
    frame_data = frame_data.dropna(subset=["frameTime"]).sort_values(
        "frameTime"
    )
    times = frame_data["frameTime"].to_numpy(dtype=float)
    mask = np.zeros(len(frame_data), dtype=bool)

    for start, end in merge_intervals(transcript):
        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, end, side="right")
        mask[left:right] = True

    columns = [
        column
        for column in frame_data.columns
        if column not in {"name", "frameTime"}
    ]
    values = frame_data.loc[mask, columns].apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def summarize_frames(frames):
    median = frames.median()
    quantiles = frames.quantile([0.10, 0.25, 0.75, 0.90])
    statistics = {
        "mean": frames.mean(),
        "std": frames.std(ddof=0),
        "median": median,
        "p10": quantiles.loc[0.10],
        "p25": quantiles.loc[0.25],
        "p75": quantiles.loc[0.75],
        "p90": quantiles.loc[0.90],
        "iqr": quantiles.loc[0.75] - quantiles.loc[0.25],
        "mad": frames.sub(median).abs().median(),
        "skew": frames.skew(),
        "kurtosis": frames.kurtosis(),
    }
    features = {}

    for statistic_name, values in statistics.items():
        for feature_name, value in values.items():
            name = "egemaps_" + feature_name + "_" + statistic_name
            features[name] = float(value) if pd.notna(value) else 0.0

    return features


def build_statistics():
    os.makedirs(statistics_folder, exist_ok=True)
    participants = read_participants()
    valid_ids = {int(row["Participant_ID"]) for row in participants}
    transcripts = find_transcripts()
    egemaps_files = find_egemaps_files()
    archives = find_archives(valid_ids)
    records = []
    completed_ids = set()

    if os.path.exists(checkpoint_file):
        records = pd.read_csv(checkpoint_file).to_dict("records")
        completed_ids = {int(row["Participant_ID"]) for row in records}
        print("Resuming after", len(completed_ids), "participants", flush=True)

    for index, participant in enumerate(participants, start=1):
        participant_id = int(participant["Participant_ID"])
        if participant_id in completed_ids:
            continue

        transcript_path = transcripts.get(participant_id)
        frames = None

        if participant_id in egemaps_files:
            try:
                frames = read_egemaps_file(egemaps_files[participant_id])
            except (OSError, pd.errors.ParserError):
                frames = None

        if frames is None and participant_id in archives:
            frames = read_egemaps_archive(participant_id, archives[participant_id])

        if transcript_path is None or frames is None:
            raise RuntimeError(
                "Missing transcript or eGeMAPS for participant "
                + str(participant_id)
            )

        transcript = pd.read_csv(transcript_path, encoding="utf-8-sig")
        participant_frames = keep_participant_frames(frames, transcript)

        if participant_frames.empty:
            raise RuntimeError(
                "No participant speech frames for " + str(participant_id)
            )

        record = dict(participant)
        record["egemaps_speech_frame_count"] = len(participant_frames)
        record.update(summarize_frames(participant_frames))
        records.append(record)
        print("Processed", index, "of", len(participants), participant_id, flush=True)

        if len(records) % 10 == 0:
            pd.DataFrame(records).to_csv(checkpoint_file, index=False)

    records.sort(key=lambda row: int(row["Participant_ID"]))
    result = pd.DataFrame(records)

    if set(result["Participant_ID"].astype(int)) != valid_ids:
        raise RuntimeError("The audio statistics table is incomplete")

    result.to_csv(statistics_file, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    feature_count = sum(
        column.startswith("egemaps_")
        and not column.endswith("speech_frame_count")
        for column in result.columns
    )
    print("Participants:", len(result), flush=True)
    print("eGeMAPS statistical features:", feature_count, flush=True)


def load_model_data():
    statistics = pd.read_csv(statistics_file)
    balanced = pd.read_csv(balanced_file)[
        ["Participant_ID", "actual_label"]
    ]
    data = balanced.merge(
        statistics,
        on="Participant_ID",
        how="left",
        validate="one_to_one",
    )

    if len(data) != 132 or data.isna().any().any():
        raise ValueError("Balanced audio data must contain all 132 participants")

    if not np.array_equal(data["actual_label"], data["label"]):
        raise ValueError("The transcript and audio labels do not match")

    feature_names = [
        column
        for column in data.columns
        if column.startswith("egemaps_")
        and not column.endswith("speech_frame_count")
    ]
    return (
        data["Participant_ID"].to_numpy(dtype=int),
        data[feature_names].to_numpy(dtype=float),
        data["label"].to_numpy(dtype=int),
        feature_names,
    )


def make_model():
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=50)),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="linear", C=0.1, gamma="scale")),
        ]
    )


def selected_features(model, feature_names):
    names_after_variance = np.asarray(feature_names)[
        model.named_steps["variance"].get_support()
    ]
    selector = model.named_steps["select"]
    names = names_after_variance[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    return [
        (str(name), float(score) if np.isfinite(score) else 0.0)
        for name, score in zip(names, scores)
    ]


def get_scores(y_values, predictions):
    matrix = confusion_matrix(y_values, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_values, predictions),
        "precision": precision_score(
            y_values, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_values, predictions, pos_label=1, zero_division=0
        ),
        "depressed_f1": f1_score(
            y_values, predictions, pos_label=1, zero_division=0
        ),
        "non_depressed_f1": f1_score(
            y_values, predictions, pos_label=0, zero_division=0
        ),
        "macro_f1": f1_score(
            y_values, predictions, average="macro", zero_division=0
        ),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def run_final_model():
    participant_ids, x_values, y_values, feature_names = load_model_data()
    predictions = []
    decision_scores = []
    selected_counts = {}
    selected_score_totals = {}

    for test_index in range(len(y_values)):
        training_mask = np.ones(len(y_values), dtype=bool)
        training_mask[test_index] = False
        model = make_model()
        model.fit(x_values[training_mask], y_values[training_mask])
        x_test = x_values[test_index].reshape(1, -1)
        predictions.append(int(model.predict(x_test)[0]))
        decision_scores.append(float(model.decision_function(x_test)[0]))

        for name, score in selected_features(model, feature_names):
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
    scores = get_scores(y_values, predictions)
    pd.DataFrame(
        {
            "Participant_ID": participant_ids,
            "actual_label": y_values,
            "predicted_label": predictions,
            "decision_score": decision_scores,
        }
    ).to_csv(prediction_file, index=False)

    selected_rows = [
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
    pd.DataFrame(selected_rows).sort_values(
        ["folds_selected", "mean_training_f_score_when_selected"],
        ascending=False,
    ).to_csv(selected_file, index=False)

    pd.DataFrame(
        [
            {
                "Method": "eGeMAPS",
                "Model": "Linear SVM",
                "Participants": len(y_values),
                "Input_features": len(feature_names),
                "Selected_features_per_fold": 50,
                "C": 0.1,
                "Accuracy": scores["accuracy"],
                "Recall": scores["recall"],
                "Precision": scores["precision"],
                "F1-score": scores["depressed_f1"],
                "Non-depressed_F1": scores["non_depressed_f1"],
                "Macro_F1": scores["macro_f1"],
                "TN": scores["tn"],
                "FP": scores["fp"],
                "FN": scores["fn"],
                "TP": scores["tp"],
            }
        ]
    ).to_csv(evaluation_file, index=False)

    print("\neGeMAPS Linear SVM - balanced LOSO", flush=True)
    print("Participants: 132 (66 non-depressed, 66 depressed)", flush=True)
    print("Accuracy:", round(scores["accuracy"] * 100, 2), "%", flush=True)
    print("Depressed F1:", round(scores["depressed_f1"], 3), flush=True)
    print("Non-depressed F1:", round(scores["non_depressed_f1"], 3), flush=True)
    print("Macro F1:", round(scores["macro_f1"], 3), flush=True)
    print(
        "Confusion matrix:",
        [[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]],
        flush=True,
    )


def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "final"

    if command == "extract":
        build_statistics()
    elif command == "final":
        if not os.path.exists(statistics_file):
            build_statistics()
        run_final_model()
    else:
        print("Use: final or extract", flush=True)


if __name__ == "__main__":
    main()
