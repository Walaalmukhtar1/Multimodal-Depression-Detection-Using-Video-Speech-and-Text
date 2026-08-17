import csv
import os
import random
import re
import sys
import tarfile
import unicodedata

project_folder = os.path.dirname(os.path.abspath(__file__))
repository_folder = os.path.dirname(project_folder)
local_packages = os.path.join(repository_folder, "packages")
if os.path.isdir(local_packages):
    sys.path.append(local_packages)

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


company_dataset_folder = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "depression project",
    "Dataset",
    "Depression Dataset",
)
local_dataset_folder = os.path.abspath(
    os.path.join(repository_folder, "data", "raw")
)
default_dataset_folder = (
    company_dataset_folder
    if os.path.isdir(company_dataset_folder)
    else local_dataset_folder
)
dataset_folder = os.environ.get(
    "EDAIC_RAW_DIR",
    os.environ.get("EDAIC_DATASET_PATH", default_dataset_folder),
)
processed_folder = os.path.join(repository_folder, "data", "processed")
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

local_nltk = os.path.join(repository_folder, "nltk_data")
if os.path.isdir(local_nltk):
    nltk.data.path.insert(0, local_nltk)

split_files = {
    "train": ["Train Split Data.csv", "train_split.csv"],
    "dev": ["Dev Split Data.csv", "dev_split.csv"],
    "test": ["Test Split Data.csv", "test_split.csv"],
}

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
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = "".join(
        " " if unicodedata.category(letter).startswith("P") else letter
        for letter in text
    )
    return re.sub(r"\s+", " ", text).strip()


def get_words(text):
    text = clean_text(text)
    return nltk.word_tokenize(text) if text else []


def average(values):
    return sum(values) / len(values) if values else 0.0


def read_labels():
    labels = []

    for split_name, file_names in split_files.items():
        path = next(
            (
                os.path.join(dataset_folder, name)
                for name in file_names
                if os.path.isfile(os.path.join(dataset_folder, name))
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                "Could not find the "
                + split_name
                + " label file in "
                + dataset_folder
                + ". Tried: "
                + ", ".join(file_names)
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


def find_archives(valid_ids):
    archive_map = {}

    for name in os.listdir(dataset_folder):
        if not name.lower().endswith((".gz", ".tgz")):
            continue

        path = os.path.join(dataset_folder, name)
        participant_id = None

        for value in re.findall(r"(?<!\d)(\d{3})(?!\d)", name):
            if int(value) in valid_ids:
                participant_id = int(value)
                break

        if participant_id is None:
            participant_id = generic_archive_ids.get(name)

        if participant_id in valid_ids:
            archive_map[participant_id] = path

    return archive_map


def extract_transcript(participant_id, archive_path):
    os.makedirs(transcript_cache, exist_ok=True)
    output_path = os.path.join(
        transcript_cache,
        str(participant_id) + "_Transcript.csv",
    )

    if os.path.exists(output_path):
        return output_path

    wanted_name = str(participant_id) + "_Transcript.csv"

    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if member.isfile() and os.path.basename(member.name).lower() == wanted_name.lower():
                    source = archive.extractfile(member)
                    if source is None:
                        return None
                    with open(output_path, "wb") as output:
                        output.write(source.read())
                    return output_path
    except (tarfile.TarError, OSError):
        return None

    return None


def read_transcript(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def extract_features(rows):
    values = {name: [] for name in feature_names}
    english_stopwords = set(stopwords.words("english")) | filler_words
    used_utterances = 0

    for row in rows:
        text = clean_text(row.get("Text", ""))
        words = get_words(text)

        if not words:
            continue

        used_utterances += 1
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
    features["utterance_count"] = used_utterances
    return features


def build_feature_table():
    os.makedirs(processed_folder, exist_ok=True)
    labels = read_labels()
    valid_ids = {row["Participant_ID"] for row in labels}
    transcript_files = find_transcripts()
    archive_map = find_archives(valid_ids)
    records = []
    missing_ids = []

    for index, label_row in enumerate(labels, start=1):
        participant_id = label_row["Participant_ID"]
        transcript_path = transcript_files.get(participant_id)

        if transcript_path is None and participant_id in archive_map:
            transcript_path = extract_transcript(
                participant_id,
                archive_map[participant_id],
            )

        if transcript_path is None:
            missing_ids.append(participant_id)
            continue

        record = dict(label_row)
        record.update(extract_features(read_transcript(transcript_path)))
        records.append(record)
        print("Processed", index, "of", len(labels), participant_id, flush=True)

    columns = [
        "Participant_ID",
        "split",
        "label",
        "phq_score",
        "utterance_count",
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

    random_generator = random.Random(42)
    selected_non_depressed = random_generator.sample(non_depressed_ids, 66)
    selected_ids = sorted(depressed_ids + selected_non_depressed)
    feature_lookup = features.set_index("Participant_ID")
    missing = [value for value in selected_ids if value not in feature_lookup.index]

    if missing:
        raise ValueError("Missing NLP features for participants: " + str(missing))

    balanced = feature_lookup.loc[selected_ids]
    x_values = balanced[feature_names].to_numpy(dtype=float)
    y_values = balanced["label"].to_numpy(dtype=int)
    return np.asarray(selected_ids), x_values, y_values


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
    os.makedirs(processed_folder, exist_ok=True)
    participant_ids, x_values, y_values = make_balanced_data()
    predictions = []
    decision_scores = []

    for test_index in range(len(y_values)):
        training_mask = np.ones(len(y_values), dtype=bool)
        training_mask[test_index] = False
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=0.01, gamma="scale"),
        )
        model.fit(x_values[training_mask], y_values[training_mask])
        x_test = x_values[test_index].reshape(1, -1)
        predictions.append(int(model.predict(x_test)[0]))
        decision_scores.append(float(model.decision_function(x_test)[0]))

    predictions = np.asarray(predictions, dtype=int)
    scores = get_scores(y_values, predictions)
    prediction_table = pd.DataFrame(
        {
            "Participant_ID": participant_ids,
            "actual_label": y_values,
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
                "Participants": len(y_values),
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

    print("\nTranscript Linear SVM - balanced LOSO", flush=True)
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
        build_feature_table()
    elif command == "final":
        if not os.path.exists(feature_file):
            build_feature_table()
        run_final_model()
    else:
        print("Use: final or extract", flush=True)


if __name__ == "__main__":
    main()
