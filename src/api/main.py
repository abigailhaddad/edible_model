from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(model_path))
from inference import load_model

app = FastAPI()
model = load_model()  # Single model instance

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):  # Remove async
    result = model.predict(input_data.text)
    print(f"Input: {input_data.text}, Result: {result}")
    return result

@app.get("/")
def root():  # Remove async
    return {"status": "alive"}