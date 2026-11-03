"""Prepare email data. Run with python -m Lab01.src.preprocessing."""

import json
import re
from html import unescape
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = LAB_DIR / "Data" / "preprocessing"


def load_data(csv_path):
    raw = pd.read_csv(csv_path, keep_default_na=False)
    if not {"text", "label_num"}.issubset(raw.columns):
        raise ValueError("CSV phải có hai cột text và label_num.")

    labels = pd.to_numeric(raw["label_num"], errors="coerce")
    valid = raw["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    valid &= labels.isin([0, 1])
    data = raw.loc[valid, ["text"]].copy()
    data.insert(0, "email_id", data.index)
    data["label_num"] = labels.loc[valid].astype(int)
    return data.reset_index(drop=True), len(raw)


def remove_html(text):
    text = unescape(text)
    text = re.sub(r"<!--.*?-->|<(script|style)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I | re.S)
    return re.sub(r"</?[a-zA-Z][\w:-]*(?:\s[^<>]*)?\s*/?>", " ", text)


def remove_punctuation(text):
    return re.sub(r"[^\w\s]|_", " ", text)


def remove_stopwords(text):
    return re.sub(r"\b\w+\b", lambda match: "" if match[0] in ENGLISH_STOP_WORDS else match[0], text)


def normalize_whitespace(text):
    return " ".join(text.split())


CLEANING_STEPS = (
    ("04_remove_html", remove_html),
    ("05_lowercase", str.lower),
    ("06_remove_punctuation", remove_punctuation),
    ("07_remove_stopwords", remove_stopwords),
    ("08_normalize_whitespace", normalize_whitespace),
)


def clean_text(text):
    """Apply the same steps used during training to a new email."""
    for _, transform in CLEANING_STEPS:
        text = transform(text)
    return text


def save_split(train, test, directory):
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)


def build_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)


def prepare_data(
    csv_path=LAB_DIR / "Data" / "raw" / "spam_ham_dataset.csv",
    result_dir=PREPROCESSING_DIR,
    model_dir=LAB_DIR / "models",
):
    """Save intermediate results and return X_train, X_test, y_train, y_test."""
    result_dir, model_dir = Path(result_dir), Path(model_dir)
    validated, original_count = load_data(csv_path)
    validated["text_clean"] = validated["text"].map(clean_text)
    nonempty = validated.loc[validated["text_clean"].ne("")]
    if nonempty.groupby("text_clean")["label_num"].nunique().gt(1).any():
        raise ValueError("Có email trùng nội dung sau làm sạch nhưng khác nhãn.")
    unique = nonempty.drop_duplicates(subset="text_clean").copy()
    if unique["label_num"].nunique() != 2:
        raise ValueError("Dữ liệu cần chứa cả ham (0) và spam (1).")

    train, test = train_test_split(
        unique, test_size=0.2, random_state=42, stratify=unique["label_num"],
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    validated.to_csv(result_dir / "01_validated.csv", index=False)
    unique.to_csv(result_dir / "02_deduplicated.csv", index=False)
    for split in (train, test):
        split.insert(1, "text_original", split["text"])
    save_split(train, test, result_dir / "03_split")

    for directory, transform in CLEANING_STEPS:
        for split in (train, test):
            split["text"] = split["text"].map(transform)
        save_split(train, test, result_dir / directory)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])
    y_train, y_test = train["label_num"].to_numpy(), test["label_num"].to_numpy()

    tfidf_dir = result_dir / "09_tfidf"
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    save_npz(tfidf_dir / "X_train.npz", X_train)
    save_npz(tfidf_dir / "X_test.npz", X_test)
    train[["email_id", "label_num"]].to_csv(tfidf_dir / "y_train.csv", index=False)
    test[["email_id", "label_num"]].to_csv(tfidf_dir / "y_test.csv", index=False)
    pd.DataFrame({"feature_name": vectorizer.get_feature_names_out()}).to_csv(
        tfidf_dir / "feature_names.csv", index=False,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")

    summary = {
        "original_rows": original_count,
        "invalid_rows_removed": original_count - len(validated),
        "empty_rows_removed": len(validated) - len(nonempty),
        "duplicates_removed": len(nonempty) - len(unique),
        "test_size": 0.2,
        "random_state": 42,
        "train_label_counts": train["label_num"].value_counts().to_dict(),
        "test_label_counts": test["label_num"].value_counts().to_dict(),
        "empty_train_after_cleaning": int(train["text"].eq("").sum()),
        "empty_test_after_cleaning": int(test["text"].eq("").sum()),
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, _, _ = prepare_data()
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")
