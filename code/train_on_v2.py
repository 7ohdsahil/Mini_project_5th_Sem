import os
import re
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# Automatically locates your uploaded CSV file
CANDIDATE_FILES = [
    "hinglish-fake-news-dataset-final-cleaned-v2_2.csv",
    "hinglish-fake-news-dataset-final-cleaned-v2.csv",
]
csv_path = next((f for f in CANDIDATE_FILES if os.path.exists(f)), None)
if csv_path is None:
    raise FileNotFoundError(f"Place one of {CANDIDATE_FILES} in the current directory.")

df = pd.read_csv(csv_path)
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"].str.len() > 0].drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=65000,
        min_df=3,
        sublinear_tf=True,
    )),
])

model_pipeline = Pipeline([
    ("features", features),
    ("clf", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=3.5, max_iter=1000, class_weight="balanced", random_state=42)),
            ("svm", CalibratedClassifierCV(
                LinearSVC(C=0.6, dual="auto", class_weight="balanced", random_state=42),
                cv=3,
                method="sigmoid",
            )),
            ("cnb", ComplementNB(alpha=0.25)),
        ],
        voting="soft",
        weights=[2, 3, 1],
    )),
])

print(f"Training ensemble on '{csv_path}' ({len(X_train)} train / {len(X_test)} test)...")
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Real News (0)", "Fake News (1)"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model_pipeline, "hinglish_fake_news_v2_model.joblib", compress=3)
print("Saved trained model -> hinglish_fake_news_v2_model.joblib")
