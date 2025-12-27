# backend/config.py

from pathlib import Path

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "backend" / "models" / "artifacts"

# Dataset paths
DISEASE_SYMPTOM_PATH = DATA_DIR / "dataset.csv"
MENDELEY_SYMPTOM_PATH = DATA_DIR / "mendeley_symptoms.csv"
SYMPTOM_DESC_PATH = DATA_DIR / "symptom_Description.csv"
SYMPTOM_PRECAUTION_PATH = DATA_DIR / "symptom_precaution.csv"
VECTOR_DB_PATH = "vectorstore/symptom_store"
# Saved model path
PREDICTOR_MODEL_PATH = MODEL_DIR / "predictor.pkl"

# Random seed for reproducibility
RANDOM_STATE = 42

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
