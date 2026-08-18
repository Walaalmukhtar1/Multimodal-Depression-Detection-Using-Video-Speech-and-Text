import csv
import os
import random
import re
import sys
import tarfile
import unicodedata

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from textblob import TextBlob


repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "depression project",
    "Dataset",
)


def find_dataset_folder():
    requested_folder = os.environ.get("EDAIC_RAW_DIR", dataset_root)
    folders_to_check = [requested_folder, dataset_root]
    train_names = {"Train Split Data.csv", "train_split.csv"}

    for starting_folder in folders_to_check:
        if not os.path.isdir(starting_folder):
            continue
        for current_folder, _, files in os.walk(starting_folder):
            if train_names.intersection(files):
                return current_folder

    return requested_folder


dataset_folder = find_dataset_folder()
processed_folder = os.environ.get(
    "EDAIC_PROCESSED_DIR",
    os.path.join(repo, "data", "processed"),
)
transcript_cache = os.path.join(processed_folder, "transcripts")
feature_file = os.path.join(processed_folder, "edaic_nlp_features.csv")
balanced_file = os.path.join(
    processed_folder,
    "edaic_nlp_balanced_loso_results.csv",
)
evaluation_file = os.path.join(
    processed_folder,
    "edaic_nlp_balanced_loso_model_evaluation.csv",
)

local_nltk = os.path.join(repo, "nltk_data")
if os.path.isdir(local_nltk):
    nltk.data.path.insert(0, local_nltk)

split_files = {
    "train": ["Train Split Data.csv", "train_split.csv"],
    "dev": ["Dev Split Data.csv", "dev_split.csv"],
    "test": ["Test Split Data.csv", "test_split.csv"],
}

feature_names = [
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

filler_words = {"uh", "um", "mm", "hmm", "erm", "ah"}
first_person_words = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = "".join(
        " " if unicodedata.category(letter).startswith("P") else letter
        for letter in text
    )
    return re.sub(r"\s+", " ", text).strip()


def average(values):
    return sum(values) / len(values) if values else 0.0


def read_labels():
    labels = []

    for split_name, possible_names in split_files.items():
        path = next(
            (
                os.path.join(dataset_folder, name)
                for name in possible_names
                if os.path.isfile(os.path.join(dataset_folder, name))
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                f"No {split_name} label file found in {dataset_folder}. "
                f"Tried: {', '.join(possible_names)}"
            )

        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                labels.append(
                    {
                        "Participant_ID": int(row["Participant_ID"]),
                        "split": split_name,
                        "label": int(row["PHQ_Binary"]),
                        "phq_score": int(row["PHQ_Score"]),
                    }
                )

    return labels


def find_transcripts():
    transcripts = {}

    for search_folder in [dataset_folder, transcript_cache]:
        if not os.path.isdir(search_folder):
            continue
        for folder, _, names in os.walk(search_folder):
            for name in names:
                match = re.fullmatch(r"(\d+)_Transcript\.csv", name, re.I)
                if match:
                    transcripts[int(match.group(1))] = os.path.join(folder, name)

    return transcripts


def participant_id_in_archive(path, valid_ids):
    for value in re.findall(r"(?<!\d)(\d{3})(?!\d)", os.path.basename(path)):
        if int(value) in valid_ids:
            return int(value)

    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                name = os.path.basename(member.name)
                match = re.fullmatch(r"(\d+)_Transcript\.csv", name, re.I)
                if match and int(match.group(1)) in valid_ids:
                    return int(match.group(1))
    except (tarfile.TarError, OSError):
        return None

    return None


def find_archives(valid_ids):
    archives = {}

    if not os.path.isdir(dataset_folder):
        raise FileNotFoundError("Dataset folder not found: " + dataset_folder)

    for name in os.listdir(dataset_folder):
        if not name.lower().endswith((".tar.gz", ".tgz", ".tar", ".gz")):
            continue
        path = os.path.join(dataset_folder, name)
        participant_id = participant_id_in_archive(path, valid_ids)
        if participant_id is not None:
            archives[participant_id] = path

    return archives


def extract_transcript(participant_id, archive_path):
    os.makedirs(transcript_cache, exist_ok=True)
    output_path = os.path.join(
        transcript_cache,
        f"{participant_id}_Transcript.csv",
    )
    if os.path.isfile(output_path):
        return output_path

    wanted = f"{participant_id}_Transcript.csv".lower()
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if os.path.basename(member.name).lower() != wanted:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    return None
                with open(output_path, "wb") as output:
                    output.write(source.read())
                return output_path
    except (tarfile.TarError, OSError):
        return None

    return None


def extract_features(rows):
    values = {name: [] for name in feature_names}
    english_stopwords = set(stopwords.words("english")) | filler_words
    utterance_count = 0

    for row in rows:
        text = clean_text(row.get("Text", ""))
        words = nltk.word_tokenize(text) if text else []
        if not words:
            continue

        utterance_count += 1
        word_count = len(words)
        values["avg_sentiment"].append(TextBlob(text).sentiment.polarity)
        values["avg_unique_frequency"].append(len(set(words)) / word_count)
        values["avg_sw_frequency"].append(
            sum(word in english_stopwords for word in words) / word_count
        )
        values["avg_characters"].append(
            sum(len(word) for word in words) / word_count
        )
        values["fp_avg"].append(
            sum(word in first_person_words for word in words) / word_count
        )

        try:
            duration = float(row.get("End_Time", 0)) - float(
                row.get("Start_Time", 0)
            )
            if duration > 0:
                values["speech_speed"].append(word_count / duration)
        except (TypeError, ValueError):
            pass

        tags = nltk.pos_tag(words)
        values["avg_nouns"].append(
            sum(tag.startswith("NN") for _, tag in tags) / word_count
        )
        values["avg_verbs"].append(
            sum(tag.startswith("VB") for _, tag in tags) / word_count
        )
        values["adj_freq"].append(
            sum(tag.startswith("JJ") for _, tag in tags) / word_count
        )
        values["avg_adv"].append(
            sum(tag.startswith("RB") for _, tag in tags) / word_count
        )

    features = {name: average(values[name]) for name in feature_names}
    features["utterance_count"] = utterance_count
    return features


def build_feature_table():
    os.makedirs(processed_folder, exist_ok=True)
    labels = read_labels()
    valid_ids = {row["Participant_ID"] for row in labels}
    transcripts = find_transcripts()
    archives = find_archives(valid_ids)
    records = []
    missing_ids = []

    for index, label_row in enumerate(labels, start=1):
        participant_id = label_row["Participant_ID"]
        transcript_path = transcripts.get(participant_id)

        if transcript_path is None and participant_id in archives:
            transcript_path = extract_transcript(
                participant_id,
                archives[participant_id],
            )
        if transcript_path is None:
            missing_ids.append(participant_id)
            continue

        with open(
            transcript_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        record = dict(label_row)
        record.update(extract_features(rows))
        records.append(record)
        print("Processed", index, "of", len(labels), participant_id, flush=True)

    columns = [
        "Participant_ID", "split", "label", "phq_score", "utterance_count"
    ] + feature_names
    pd.DataFrame(records)[columns].to_csv(feature_file, index=False)
    print("Feature rows:", len(records), flush=True)
    print("Missing participants:", missing_ids, flush=True)


def make_balanced_data():
    features = pd.read_csv(feature_file)
    labels = read_labels()
    depressed_ids = [
        row["Participant_ID"] for row in labels if row["label"] == 1
    ]
    non_depressed_ids = [
        row["Participant_ID"] for row in labels if row["label"] == 0
    ]

    if len(depressed_ids) != 66:
        raise ValueError("Expected 66 depressed participants")

    selected_non_depressed = random.Random(42).sample(non_depressed_ids, 66)
    selected_ids = sorted(depressed_ids + selected_non_depressed)
    feature_lookup = features.set_index("Participant_ID")
    missing = [pid for pid in selected_ids if pid not in feature_lookup.index]
    if missing:
        raise ValueError("Missing NLP features for participants: " + str(missing))

    balanced = feature_lookup.loc[selected_ids]
    return (
        np.asarray(selected_ids),
        balanced[feature_names].to_numpy(dtype=float),
        balanced["label"].to_numpy(dtype=int),
    )


def calculate_scores(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "depressed_f1": f1_score(labels, predictions, zero_division=0),
        "non_depressed_f1": f1_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def run_model():
    participant_ids, features, labels = make_balanced_data()
    predictions = []
    decision_scores = []

    for test_index in range(len(labels)):
        train = np.ones(len(labels), dtype=bool)
        train[test_index] = False
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=0.01),
        )
        model.fit(features[train], labels[train])
        test_row = features[test_index].reshape(1, -1)
        predictions.append(int(model.predict(test_row)[0]))
        decision_scores.append(float(model.decision_function(test_row)[0]))

    predictions = np.asarray(predictions)
    scores = calculate_scores(labels, predictions)
    prediction_table = pd.DataFrame(
        {
            "Participant_ID": participant_ids,
            "actual_label": labels,
            "predicted_label": predictions,
            "decision_score": decision_scores,
        }
    )
    for name, value in scores.items():
        prediction_table["overall_" + name] = value
    prediction_table.to_csv(balanced_file, index=False)

    pd.DataFrame(
        [
            {
                "Model": "Linear SVM",
                "Participants": len(labels),
                "Features": len(feature_names),
                "C": 0.01,
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

    print("\nTranscript Linear SVM - balanced LOSO")
    print("Participants: 132 (66 non-depressed, 66 depressed)")
    print("Accuracy:", round(scores["accuracy"] * 100, 2), "%")
    print("Depressed F1:", round(scores["depressed_f1"], 3))
    print("Non-depressed F1:", round(scores["non_depressed_f1"], 3))
    print("Macro F1:", round(scores["macro_f1"], 3))
    print(
        "Confusion matrix:",
        [[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]],
    )


def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "final"
    print("Dataset folder:", dataset_folder)

    if command == "extract":
        build_feature_table()
    elif command == "final":
        if not os.path.isfile(feature_file):
            build_feature_table()
        run_model()
    else:
        print("Use: final or extract")


if __name__ == "__main__":
    main()
