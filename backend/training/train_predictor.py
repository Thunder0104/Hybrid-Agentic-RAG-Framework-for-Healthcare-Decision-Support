# backend/training/train_predictor.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import random
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from backend.config import DISEASE_SYMPTOM_PATH, PREDICTOR_MODEL_PATH, RANDOM_STATE

# -------------------------
# OPTIONAL: introduce noise
# -------------------------
def add_noise(symptom_list, all_symptoms):
    """Randomly drop, duplicate, or add a symptom to make model more realistic."""
    noisy = symptom_list.copy()

    # 10% chance to drop one symptom
    if random.random() < 0.10 and len(noisy) > 1:
        drop_idx = random.randint(0, len(noisy) - 1)
        noisy.pop(drop_idx)

    # 10% chance to add a random new symptom
    if random.random() < 0.10:
        new_sym = random.choice(all_symptoms)
        if new_sym not in noisy:
            noisy.append(new_sym)

    # 5% chance to duplicate a symptom (doesn't change much)
    if random.random() < 0.05 and len(noisy) > 0:
        noisy.append(random.choice(noisy))

    # remove duplicates, sort for consistency
    noisy = sorted(list(set(noisy)))
    return noisy


def train_predictor():
    df = pd.read_csv(DISEASE_SYMPTOM_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    # Columns Symptom_1 .. Symptom_17
    symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
    df[symptom_cols] = df[symptom_cols].fillna("")

    # Normalize symptom names
    df["Symptoms"] = df[symptom_cols].apply(
        lambda row: [s.strip().lower().replace(" ", "_") for s in row if s != ""],
        axis=1
    )

    # collect the full symptom universe
    all_symptoms = sorted({s for sub in df["Symptoms"] for s in sub})

    # introduce noise (controlled)
    df["Symptoms_noisy"] = df["Symptoms"].apply(
        lambda syms: add_noise(syms, all_symptoms)
    )

    df = df[["Disease", "Symptoms_noisy"]].rename(columns={"Symptoms_noisy": "Symptoms"})

    # Encode
    mlb = MultiLabelBinarizer()
    label_enc = LabelEncoder()

    X_train, X_test, y_train, y_test = train_test_split(
        df["Symptoms"], df["Disease"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["Disease"]
    )

    X_train_enc = pd.DataFrame(mlb.fit_transform(X_train), columns=mlb.classes_)
    X_test_enc = pd.DataFrame(mlb.transform(X_test), columns=mlb.classes_)

    print(X_train_enc)
    y_train_enc = label_enc.fit_transform(y_train)
    y_test_enc = label_enc.transform(y_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    # clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # clf = SVC(probability=True)

    clf.fit(X_train_enc, y_train_enc)

    # Evaluate
    y_pred = clf.predict(X_test_enc)
    train_proba = clf.predict_proba(X_train_enc)
    y_proba = clf.predict_proba(X_test_enc)

    acc = accuracy_score(y_test_enc, y_pred)
    train_auc = roc_auc_score(y_train_enc, train_proba, multi_class='ovr')
    auc = roc_auc_score(y_test_enc, y_proba, multi_class="ovr")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Training AUC : {train_auc:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    joblib.dump(
        {
            "model": clf,
            "mlb": mlb,
            "label_encoder": label_enc,
        },
        PREDICTOR_MODEL_PATH
    )
    print(f"Model saved to: {PREDICTOR_MODEL_PATH}")


if __name__ == "__main__":
    train_predictor()
