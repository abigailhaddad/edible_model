from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(model_path))
from inference import load_model

app = FastAPI()
model = load_model()

# Replace with your GitHub Pages domain
ALLOWED_ORIGINS = [
    "https://abigailhaddad.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Specify only the methods you need
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    result = model.predict(input_data.text)
    print(f"Input: {input_data.text}, Result: {result}")
    return result

@app.get("/")
def root():
    return {"status": "alive"}