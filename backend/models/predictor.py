# backend/models/predictor.py

import joblib
import numpy as np
from pathlib import Path

# path to the saved model
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "predictor.pkl"

class DiseasePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        # load model + encoders
        model_bundle = joblib.load(model_path)
        self.model = model_bundle["model"]
        self.mlb = model_bundle["mlb"]
        self.label_encoder = model_bundle["label_encoder"]

    def predict(self, symptoms):
        """
        Predict disease(s) for a given list of symptoms.
        Example:
            predictor = DiseasePredictor()
            predictor.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
        """

        # normalize inputs to match training
        symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms]

        # ensure all features exist in the same order as training
        X_input = self.mlb.transform([symptoms])  # returns a 1×N feature vector

        # get predicted probabilities
        y_proba = self.model.predict_proba(X_input)[0]
        y_pred_idx = np.argmax(y_proba)
        y_pred_label = self.label_encoder.inverse_transform([y_pred_idx])[0]

        # sort all diseases by probability
        sorted_indices = np.argsort(y_proba)[::-1]
        top_predictions = [
            {
                "disease": self.label_encoder.inverse_transform([i])[0],
                "probability": float(y_proba[i]),
            }
            for i in sorted_indices[:5]  # top 5 results
        ]

        return {
            "predicted_disease": y_pred_label,
            "confidence": float(y_proba[y_pred_idx]),
            "top_predictions": top_predictions,
        }


# quick test (if you run this file directly)
if __name__ == "__main__":
    predictor = DiseasePredictor()
    # result = predictor.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
    result = predictor.predict(["weight_loss", "nausea", "muscle_pain"])
    print(result)
