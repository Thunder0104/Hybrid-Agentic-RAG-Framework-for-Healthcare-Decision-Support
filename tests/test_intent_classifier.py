# backend/models/test_intent_classifier.py

from backend.models.intent_classifier import IntentClassifier

def test():
    clf = IntentClassifier()

    tests = [
        "I have fever and body pain.",
        "What is malaria?",
        "Explain how insulin works",
        "My throat is sore and I have chills",
        "Causes of nausea?",
        "I am feeling under the weather"
    ]

    for q in tests:
        print("\nQuery:", q)
        print("Intent:", clf.classify(q))


if __name__ == "__main__":
    test()
