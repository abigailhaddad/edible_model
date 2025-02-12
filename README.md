#  Is It Edible?

A machine learning project that fine-tunes ModernBERT for classifying items as edible or non-edible. The model is deployed via FastAPI on PythonAnywhere with a JavaScript frontend.


## Project Structure

```
EDIBLE_MODEL/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI server implementation
│   │   └── test.py
│   ├── frontend/
│   ├── model/
│   │   ├── inference.py     # Model inference class
│   │   ├── train.py         # Training pipeline
│   │   └── test.py
│   └── synthetic_data/
│       └── data.json        # Training data
├── requirements.txt         # Python dependencies
├── index.html              # Frontend interface
├── styles.css              # Frontend styling
└── script.js               # Frontend logic
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abigailhaddad/edible_model
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies from `requirements.txt`

## Model Details

The model is based on ModernBERT-base and fine-tuned for binary classification:
- Uses the `answerdotai/ModernBERT-base` checkpoint
- Training includes evaluation metrics (accuracy, F1 score)
- Confusion matrix analysis with example tracking

### Training Process

The training pipeline (`train.py`) includes:
- Data preprocessing and tokenization
- Train/test split with 80/20 ratio

## API Usage

The model is served via FastAPI with the following endpoints:

### Prediction Endpoint
```python
POST /predict
```
Request body:
```json
{
    "text": "your item description"
}
```

Response:
```json
{
    "prediction": "edible" | "not_edible",
    "confidence": float,
    "probabilities": {
        "edible": float,
        "not_edible": float
    }
}
```

## Web Interface

The model can be accessed through a web interface at: https://abigailhaddad.github.io/edible_model/

### Features
- Simple text input for item descriptions
- Real-time prediction results
- Visual feedback with confidence levels:
  - Green checkmark for high-confidence edible predictions (>80%)
  - Red X for high-confidence non-edible predictions (>80%)
  - Yellow question mark for uncertain predictions (<80%)
  - Error icon with detailed message for API issues

### User Interaction
- Submit predictions by clicking "Check" or pressing Enter
- Disabled button state during prediction to prevent double submissions
- Clear visual feedback about prediction status
- Error handling with informative messages
- Character count feedback for long inputs

### API Integration
- Connects to PythonAnywhere endpoint (https://abigailhaddad1.pythonanywhere.com/predict)
- Handles network errors gracefully
- Provides detailed feedback for API failures
- Implements proper error state management

## Limitations

- The model is not particularly accurate! You should not decide what to eat based on this model's predictions.


## License

MIT license