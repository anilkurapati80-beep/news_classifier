"""Load real AG News data (or a labeled CSV) and train TF-IDF + MultinomialNB."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

AG_NEWS_SOURCE = "https://huggingface.co/datasets/fancyzhx/ag_news"
AG_NEWS_NAME = "fancyzhx/ag_news"
AG_NEWS_CATEGORIES = ("World", "Sports", "Business", "Sci/Tech")


def load_ag_news(cache_dir: Path, csv_output: Path) -> pd.DataFrame:
    """Load the official AG News training split and export validated rows to CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set this before importing Hugging Face libraries so their cache constants
    # resolve inside the project rather than the user's profile.
    os.environ["HF_HOME"] = str((cache_dir.parent / "huggingface-home").resolve())
    try:
        from datasets import ClassLabel, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for --source ag-news. "
            "Install project requirements with: python -m pip install -r requirements.txt"
        ) from exc
    dataset = load_dataset(AG_NEWS_NAME, split="train", cache_dir=str(cache_dir))
    required = {"text", "label"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"Unexpected AG News columns {dataset.column_names}; expected at least {sorted(required)}."
        )
    label_feature = dataset.features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise ValueError(f"Expected AG News 'label' to be ClassLabel, got {label_feature!r}.")
    label_names = list(label_feature.names)
    if set(label_names) != set(AG_NEWS_CATEGORIES) or len(label_names) != 4:
        raise ValueError(
            f"Unexpected AG News label mapping: {label_names}. "
            f"Expected the four categories {list(AG_NEWS_CATEGORIES)}."
        )
    print(f"AG News label ID mapping: {dict(enumerate(label_names))}")

    raw = dataset.to_pandas()[["text", "label"]].copy()
    raw["category"] = raw["label"].map(lambda value: label_names[int(value)] if pd.notna(value) else None)
    frame = raw.rename(columns={"text": "text"})[["text", "category"]]
    frame = frame.dropna(subset=["text", "category"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["category"] != "")]
    before_deduplication = len(frame)
    frame = frame.drop_duplicates(subset=["text", "category"]).copy()
    if set(frame["category"].unique()) != set(AG_NEWS_CATEGORIES):
        raise ValueError(
            f"Usable AG News rows do not contain all expected categories: "
            f"{sorted(frame['category'].unique())}"
        )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False, encoding="utf-8")
    print(f"Source: {AG_NEWS_SOURCE} (only the official train split is passed to modeling)")
    print(f"Rows received: {len(raw)}; invalid rows removed: {len(raw) - before_deduplication}; "
          f"duplicate text/category rows removed: {before_deduplication - len(frame)}")
    return frame


def train(data: Path, text_column: str, label_column: str, output: Path,
          results_dir: Path, test_size: float = 0.2, random_state: int = 42,
          source: str = "csv", cache_dir: Path = Path(".cache/huggingface"),
          dataset_name: str | None = None):
    dataset_source = AG_NEWS_SOURCE if source == "ag-news" else (dataset_name or str(data))
    if source == "ag-news":
        frame = load_ag_news(cache_dir, data)
        text_column, label_column = "text", "category"
    elif source == "csv":
        if not data.is_file():
            raise FileNotFoundError(f"Dataset not found: {data}. Supply a labeled CSV; see README.md.")
        frame = pd.read_csv(data)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Available columns: {list(frame.columns)}")
    frame = frame[[text_column, label_column]].dropna().copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame[label_column] = frame[label_column].astype(str).str.strip()
    frame = frame[(frame[text_column] != "") & (frame[label_column] != "")]
    before_deduplication = len(frame)
    frame = frame.drop_duplicates(subset=[text_column, label_column]).copy()
    print(f"Valid records after blank/missing rows: {before_deduplication}")
    print(f"Exact duplicate text/category rows removed: {before_deduplication - len(frame)}")
    print(f"Records available for modeling: {len(frame)}")
    print(f"Categories available: {sorted(frame[label_column].unique())}")
    if len(frame) < 2 or frame[label_column].nunique() < 2:
        raise ValueError("Need at least two valid rows and two distinct categories.")
    print("Rows per category (after invalid rows removed):")
    print(frame[label_column].value_counts().to_string())

    class_count = frame[label_column].nunique()
    test_rows = int(math.ceil(len(frame) * test_size))
    train_rows = len(frame) - test_rows
    can_stratify = (
        frame[label_column].value_counts().min() >= 2
        and test_rows >= class_count
        and train_rows >= class_count
    )
    stratify = frame[label_column] if can_stratify else None
    if stratify is None:
        print("Warning: at least one category has fewer than two rows; using an unstratified split.")
    train_x, test_x, train_y, test_y = train_test_split(
        frame[text_column], frame[label_column], test_size=test_size,
        random_state=random_state, stratify=stratify)
    if train_y.nunique() < 2:
        raise ValueError("Training split has fewer than two categories; provide more balanced labeled data.")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000,
                                  sublinear_tf=True, stop_words="english")),
        ("classifier", MultinomialNB()),
    ])
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    labels = list(AG_NEWS_CATEGORIES) if source == "ag-news" else sorted(frame[label_column].unique())
    accuracy = accuracy_score(test_y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, predictions, average="weighted", zero_division=0)
    report = classification_report(test_y, predictions, labels=labels,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(test_y, predictions, labels=labels).tolist()
    results = {
        "dataset": dataset_source, "dataset_file": str(data), "rows_after_cleaning": len(frame),
        "train_rows": len(train_x), "test_rows": len(test_x),
        "category_distribution": {
            str(category): int(count)
            for category, count in frame[label_column].value_counts().items()
        },
        "categories": labels, "split_random_state": random_state,
        "split_stratified": stratify is not None,
        "test_metrics": {"accuracy": accuracy, "weighted_precision": precision,
                         "weighted_recall": recall, "weighted_f1": f1},
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "values": matrix},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "evaluation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(pipeline, output)
    print(classification_report(test_y, predictions, zero_division=0))
    print("Confusion matrix (category order:", labels, ")")
    print(matrix)
    print("Held-out test metrics:", results["test_metrics"])
    print(f"Saved pipeline to {output} and evaluation to {results_dir / 'evaluation.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("ag-news", "csv"), default="csv")
    parser.add_argument("--dataset-name", default=None,
                        help="Name or URL recorded in results when using --source csv.")
    parser.add_argument("--data", type=Path, default=Path("data/news.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="category")
    parser.add_argument("--output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")
    train(args.data, args.text_column, args.label_column, args.output,
          args.results_dir, args.test_size, args.random_state, args.source, args.cache_dir,
          args.dataset_name)


if __name__ == "__main__":
    main()
