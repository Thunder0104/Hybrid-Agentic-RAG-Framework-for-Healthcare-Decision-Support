# backend/training/train_predictor.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import random
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from backend.config import DISEASE_SYMPTOM_PATH, MENDELEY_SYMPTOM_PATH, PREDICTOR_MODEL_PATH, RANDOM_STATE

# -------------------------
# OPTIONAL: introduce noise
# -------------------------
def add_noise(symptom_list, all_symptoms):
    """Randomly drop, duplicate, or add a symptom to make model more realistic."""
    noisy = list(symptom_list)

    # 1) Drop 40–60% of existing symptoms
    if len(noisy) > 0:
        kept = []
        for s in noisy:
            # keep with ~50% chance
            if random.random() > 0.4:
                kept.append(s)
        noisy = kept

        # Ensure at least 1 symptom remains if original wasn't empty
        if not noisy:
            noisy = [random.choice(symptom_list)]

    # 2) Add 0–2 random extra symptoms (20% chance each)
    for _ in range(2):
        if random.random() < 0.2 and len(all_symptoms) > 0:
            new_sym = random.choice(all_symptoms)
            if new_sym not in noisy:
                noisy.append(new_sym)

    # remove duplicates, sort for consistency
    noisy = sorted(list(set(noisy)))
    return noisy


def train_predictor():
    df = pd.read_csv(DISEASE_SYMPTOM_PATH)
    print(df.head())
    print(0)
    df_m = pd.read_csv(MENDELEY_SYMPTOM_PATH)

# All columns except the last one are symptoms; last one is the label
    symptom_cols = df_m.columns[:-1]
    label_col = "prognosis"   # last column name in mendeley_symptoms.csv

    rows = []
    max_symptoms = 15  # to mimic dataset.csv (Symptom_1 ... Symptom_15)

    for _, row in df_m.iterrows():
        disease = row[label_col]

        # Get list of symptoms that are 'on' (== 1) for this row
        active_symptoms = [sym for sym in symptom_cols if row[sym] == 1]

        # Build a record in dataset.csv style
        rec = {"Disease": disease}
        for i in range(17):
            if i < len(active_symptoms):
                # Optional: make symptom names prettier
                rec[f"Symptom_{i+1}"] = active_symptoms[i].replace("_", " ")
            else:
                rec[f"Symptom_{i+1}"] = float('nan')

        rows.append(rec)

    # New DataFrame in the same shape/style as dataset.csv
    df_m_simple = pd.DataFrame(rows)

    # Preview
    print(df_m_simple.head())
    print(1)

    df = df_m_simple
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
    print(df.head())
    # Encode
    mlb = MultiLabelBinarizer()
    label_enc = LabelEncoder()

    X_all = df["Symptoms"]
    y_all = df["Disease"]

    X_all_enc = pd.DataFrame(mlb.fit_transform(X_all), columns=mlb.classes_)
    y_all_enc = label_enc.fit_transform(y_all)

    # 5-fold Stratified CV -> each fold ≈ 80% train / 20% validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_enc, y_all_enc), start=1):
        X_train, X_val = X_all_enc.iloc[train_idx], X_all_enc.iloc[val_idx]
        y_train, y_val = y_all_enc[train_idx], y_all_enc[val_idx]

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        fold_acc = accuracy_score(y_val, y_pred)
        fold_auc = roc_auc_score(y_val, y_proba, multi_class="ovr")

        acc_scores.append(fold_acc)
        auc_scores.append(fold_auc)

        print(f"Fold {fold} - Accuracy: {fold_acc:.4f}, AUC: {fold_auc:.4f}")

    print("==== Cross-validation results (5-fold, ~80/20 each) ====")
    print(f"Mean Accuracy: {sum(acc_scores)/len(acc_scores):.4f}")
    print(f"Mean AUC     : {sum(auc_scores)/len(auc_scores):.4f}")

    # Train final model on FULL data
    final_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    final_clf.fit(X_all_enc, y_all_enc)

    joblib.dump(
        {
            "model": final_clf,
            "mlb": mlb,
            "label_encoder": label_enc,
        },
        PREDICTOR_MODEL_PATH
    )
    print(f"Final model trained on full data and saved to: {PREDICTOR_MODEL_PATH}")


if __name__ == "__main__":
    train_predictor()
