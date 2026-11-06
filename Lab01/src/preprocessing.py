import argparse
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
DATA_DIR = LAB_DIR / "Data"
PREPROCESSING_DIR = DATA_DIR / "preprocessing"


def clean_text(text):
    text = unescape(text)
    text = re.sub(r"<!--.*?-->|<(script|style)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"</?[a-zA-Z][\w:-]*(?:\s[^<>]*)?\s*/?>", " ", text)
    text = re.sub(r"[^\w\s]|_", " ", text.lower())
    return " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)


def main(output_dir=PREPROCESSING_DIR):
    raw = pd.read_csv(DATA_DIR / "raw/spam_ham_dataset.csv", keep_default_na=False)
    labels = pd.to_numeric(raw["label_num"], errors="coerce")
    valid = raw["text"].map(lambda text: isinstance(text, str) and bool(text.strip()))
    valid &= labels.isin([0, 1])
    cleaned = raw.loc[valid, ["text"]].copy()
    cleaned.insert(0, "email_id", cleaned.index)
    cleaned["label_num"] = labels.loc[valid].astype(int)
    cleaned["text_clean"] = cleaned["text"].map(clean_text)
    nonempty = cleaned.loc[cleaned["text_clean"].ne("")]
    if nonempty.groupby("text_clean")["label_num"].nunique().gt(1).any():
        raise ValueError("Trùng văn bản sạch nhưng khác nhãn.")
    unique = nonempty.drop_duplicates(subset="text_clean")

    target = output_dir / "cleaned_raw.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(target, index=False)
    print(f"raw={len(raw)} invalid={len(raw)-len(cleaned)} "
          f"empty_after_cleaning={len(cleaned)-len(nonempty)} "
          f"duplicates={len(nonempty)-len(unique)} remaining={len(unique)}")
    summary = {
        "original_rows": len(raw),
        "invalid_rows_removed": len(raw) - len(cleaned),
        "empty_rows_removed": len(cleaned) - len(nonempty),
        "duplicates_removed": len(nonempty) - len(unique),
        "clean_rows": len(unique),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(unique["label_num"].value_counts().sort_index().to_dict())
    print("Example:", clean_text("<b>FREE!!!</b> Click now for the prize."))


def split_clean_data(output_dir=PREPROCESSING_DIR):
    cleaned = pd.read_csv(output_dir / "cleaned_raw.csv", keep_default_na=False)
    X = cleaned["text_clean"]
    y = cleaned["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    # Cùng chỉ số dòng giữ văn bản, nhãn và email_id đi cùng nhau.
    for name, features, labels in (
        ("train", X_train, y_train),
        ("test", X_test, y_test),
    ):
        split = pd.DataFrame({
            "email_id": cleaned.loc[features.index, "email_id"],
            "text_clean": features,
            "label_num": labels,
        })
        split.to_csv(output_dir / f"{name}_clean.csv", index=False)
        print(f"{name}: {len(split)} emails, {labels.value_counts().sort_index().to_dict()}")


def build_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)


def vectorize_clean_splits(output_dir=PREPROCESSING_DIR):
    train = pd.read_csv(output_dir / "train_clean.csv", keep_default_na=False)
    test = pd.read_csv(output_dir / "test_clean.csv", keep_default_na=False)

    vectorizer = build_vectorizer()
    # Chỉ học từ vựng và IDF từ train. Test dùng lại các cột đã học.
    X_train = vectorizer.fit_transform(train["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])

    output = output_dir / "09_tfidf"
    output.mkdir(exist_ok=True)
    for name, features, rows in (("train", X_train, train), ("test", X_test, test)):
        save_npz(output / f"X_{name}.npz", features)
        rows[["email_id", "label_num"]].to_csv(output / f"y_{name}.csv", index=False)
        print(f"{name}: {features.shape}, {features.nnz} nonzero values")
    pd.DataFrame({"feature_name": vectorizer.get_feature_names_out()}).to_csv(
        output / "feature_names.csv", index=False,
    )
    joblib.dump(vectorizer, output / "tfidf_vectorizer.joblib")
    return X_train, X_test, train["label_num"].to_numpy(), test["label_num"].to_numpy()


def prepare_data(output_dir=PREPROCESSING_DIR):
    main(output_dir)
    split_clean_data(output_dir)
    return vectorize_clean_splits(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Lab01 email data.")
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--tfidf", action="store_true", help="Vectorize the clean splits.")
    steps.add_argument("--split", action="store_true", help="Split the cleaned CSV.")
    args = parser.parse_args()
    if args.tfidf:
        vectorize_clean_splits()
    elif args.split:
        split_clean_data()
    else:
        main()
