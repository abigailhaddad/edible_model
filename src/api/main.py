# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add model directory to path
model_path = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(model_path))

from inference import EdibilityClassifier

app = FastAPI()
model = EdibilityClassifier()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    return model.predict(input_data.text)

@app.get("/")
async def root():
    return {"status": "alive"}