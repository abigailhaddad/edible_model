from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdibilityClassifier:
    def __init__(self):
        model_dir = Path(__file__).resolve().parent / "saved_model"
        logger.info(f"Attempting to load model from: {model_dir}")
        logger.info(f"Directory exists: {model_dir.exists()}")
        if model_dir.exists():
            logger.info(f"Directory contents: {list(model_dir.glob('*'))}")
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            logger.info("Successfully loaded model")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            logger.info("Successfully loaded tokenizer")
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

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
    logger.info("Starting model loading process")
    try:
        model = EdibilityClassifier()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise