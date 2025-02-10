from inference import load_model

model = load_model()
test_texts = [
    "fresh apple",
    "cardboard box",
    "portebello mushroom",
    "plastic bag",
    "human flesh",
    "chicken thight"
]

for text in test_texts:
    result = model.predict(text)
    print(f"\nText: {text}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")