from fastapi import FastAPI, HTTPException
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
async def predict(input_data: TextInput):
    # Check input length
    if len(input_data.text) > 30:
        raise HTTPException(
            status_code=400,
            detail="Input text is too long. Please keep it under 30 characters."
        )

    result = model.predict(input_data.text)
    print(f"Input: {input_data.text}, Result: {result}")
    return result

@app.get("/")
async def root():
    return {"status": "alive"}