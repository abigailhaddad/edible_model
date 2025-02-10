from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

class EdibilityClassifier:
    def __init__(self):
        model_dir = Path(__file__).resolve().parent / "saved_model"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = softmax(outputs.logits.numpy(), axis=1)
        prediction = probabilities.argmax(axis=1)[0]
        confidence = probabilities[0][prediction]
        
        return {
            "prediction": "edible" if prediction == 1 else "not_edible",
            "confidence": float(confidence),
            "probabilities": {
                "edible": float(probabilities[0][1]),
                "not_edible": float(probabilities[0][0])
            }
        }

def load_model():
    return EdibilityClassifier()

