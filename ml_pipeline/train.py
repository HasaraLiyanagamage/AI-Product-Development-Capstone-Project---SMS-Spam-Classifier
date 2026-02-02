import json
import os
import warnings
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKEND_MODEL_DIR = PROJECT_ROOT / "backend" / "model"
MODEL_PATH = BACKEND_MODEL_DIR / "sms_spam_model.joblib"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
CONFUSION_PNG = ARTIFACTS_DIR / "confusion_matrix.png"

DATA_URLS = [
    # TSV with label + text
    "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv",
    # Classic SMSSpamCollection (tab-separated, no header)
    "https://raw.githubusercontent.com/naiveHobo/SMSSpamCollection/master/SMSSpamCollection",
]

FALLBACK_DATA = [
    ("ham", "I'll call you later when I get off work."),
    ("ham", "Are we still meeting for lunch?"),
    ("ham", "Don't forget to bring the documents."),
    ("ham", "Can you send me the report by today?"),
    ("ham", "See you at the office."),
    ("ham", "Happy birthday! Have a great day!"),
    ("ham", "Let's catch up soon."),
    ("ham", "Your appointment is scheduled for 3pm."),
    ("ham", "Please review the attached file."),
    ("ham", "Thanks for your help."),
    ("spam", "Congratulations! You have won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now."),
    ("spam", "URGENT! Your account has been compromised. Verify at http://secure-verify.com"),
    ("spam", "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 12345 now!"),
    ("spam", "Earn money from home!!! Click here to find out how."),
    ("spam", "You have been selected for a prize. Call now!"),
    ("spam", "Claim your free vacation to Bahamas by responding YES."),
    ("spam", "Win a brand new iPhone. Limited time offer."),
    ("spam", "Get cheap meds without prescription. Visit our site."),
    ("spam", "Lowest car insurance rates guaranteed. Reply QUOTE."),
    ("spam", "Act now to secure your free trial subscription."),
]


def _read_local(data_path: Path) -> Optional[pd.DataFrame]:
    try:
        if not data_path.exists():
            return None
        # Try headerless first
        df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
        # Drop stray header rows
        df = df[df["label"].astype(str).str.lower().isin(["ham", "spam", "v1"]) == True]
        # Convert v1/v2 style if present
        df.loc[df["label"].str.lower() == "v1", "label"] = "ham"
        df.loc[df["label"].str.lower() == "v2", "label"] = "spam"
        df = df[df["label"].isin(["ham", "spam"])]
        df = df.rename(columns={0: "label", 1: "text"})
        return df[["label", "text"]].dropna()
    except Exception:
        return None


def _read_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        s = StringIO(r.text)
        df = pd.read_csv(s, sep="\t", header=None, names=["label", "text"], encoding="utf-8", engine="python")
        # Handle header-like first row
        if str(df.iloc[0, 0]).lower() in {"label", "v1"}:
            df = df.iloc[1:].reset_index(drop=True)
        df.loc[df["label"].astype(str).str.lower() == "v1", "label"] = "ham"
        df.loc[df["label"].astype(str).str.lower() == "v2", "label"] = "spam"
        df = df[df["label"].isin(["ham", "spam"])]
        return df[["label", "text"]].dropna()
    except Exception:
        return None


def load_dataset(prefer_online: bool = True, local_path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from URL or local path; fall back to small in-memory sample."""
    # Local first if explicitly given
    if local_path:
        maybe = _read_local(Path(local_path))
        if maybe is not None and not maybe.empty:
            return maybe

    # Try online sources
    if prefer_online:
        for url in DATA_URLS:
            df = _read_from_url(url)
            if df is not None and not df.empty:
                return df

    # Fallback small sample
    return pd.DataFrame(FALLBACK_DATA, columns=["label", "text"]).sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)),
    ])


def evaluate_and_save_artifacts(y_true: pd.Series, y_pred: np.ndarray, labels: Tuple[str, str] = ("ham", "spam")) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=list(labels), output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    metrics = {"accuracy": acc, "classification_report": report}
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG)
    plt.close()


def train_and_save(output_model_path: Optional[Path] = None, dataset_path: Optional[str] = None) -> Path:
    df = load_dataset(prefer_online=True, local_path=dataset_path)
    df = df.dropna().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    # Ensure labels are strings 'ham'/'spam'
    df["label"] = df["label"].astype(str).str.lower().map({"ham": "ham", "spam": "spam"})
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"] if df["label"].nunique() > 1 else None
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    evaluate_and_save_artifacts(y_test, y_pred, labels=("ham", "spam"))

    model_path = output_model_path or MODEL_PATH
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    return model_path


if __name__ == "__main__":
    path = train_and_save()
    print(f"Model saved to: {path}")
