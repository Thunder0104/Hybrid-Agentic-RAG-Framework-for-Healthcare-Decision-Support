# backend/training/train_predictor_mendeley.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from backend.config import DATA_DIR, PREDICTOR_MODEL_PATH, RANDOM_STATE

MENDELEY_DATA_PATH = DATA_DIR / "mendeley_symptoms.csv"

def train_mendeley_predictor():
    # 1. Load the dataset with headers
    df = pd.read_csv(MENDELEY_DATA_PATH)
    print(f"Loaded Mendeley dataset with shape: {df.shape}")

    # 2. Separate features and label
    X = df.iloc[:, :-1]        # all symptom columns
    y = df.iloc[:, -1]         # last column = prognosis

    # 3. Encode the labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # 5. Train the model
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 6. Predictions and probabilities
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # 7. Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    # 8. Save model and encoders
    joblib.dump(
        {
            "model": clf,
            "label_encoder": le,
            "columns": X.columns.tolist(),
        },
        PREDICTOR_MODEL_PATH
    )
    print(f"Model saved to: {PREDICTOR_MODEL_PATH}")

if __name__ == "__main__":
    train_mendeley_predictor()
