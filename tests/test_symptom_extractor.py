# backend/models/test_symptom_extractor.py

from backend.models.symptom_extractor import SymptomExtractor

def test():
    extractor = SymptomExtractor()

    tests = [
        "I have been losing weight lately and feel nauseous.",
        "Fever, chills and muscle pain after a trip.",
        "My throat hurts and I have mild cough.",
        "Headache and dizziness after running.",
        "I am feeling under the weather"
    ]

    for q in tests:
        print("\nQuery:", q)
        symptoms = extractor.extract(q)
        print("Extracted symptoms:", symptoms)


if __name__ == "__main__":
    test()
