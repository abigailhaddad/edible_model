import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import mlflow
from prefect import flow, task
import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding)
from scipy.special import softmax
from sklearn.metrics import f1_score, classification_report

# Set up logging with a more focused configuration
def setup_logging():
    model_dir = Path(__file__).resolve().parent
    log_file_path = model_dir / "results" / "training.log"
    
    # Configure root logger to ERROR to suppress most external library logs
    logging.getLogger().setLevel(logging.ERROR)
    
    # Create our application logger
    logger = logging.getLogger("model_training")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add only the file handler - let Prefect handle console output
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Suppress unnecessary warnings and info messages
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

@task
def setup_directories():
    """Ensure required directories exist."""
    model_dir = Path(__file__).resolve().parent
    paths = [model_dir / "synthetic_data", model_dir / "results", model_dir / "checkpoints"]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return model_dir

@task
def create_dataframe():
    """Loads the JSON file containing labeled data and returns a pandas DataFrame."""
    model_dir = Path(__file__).resolve().parent
    src_dir = model_dir.parent
    data_path = src_dir / "model" / "synthetic_data" / "data.json"
    
    with open(data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["labeled_data"])
    return df

@task
def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)

@task
def create_hf_datasets(train_df, test_df):
    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "test": datasets.Dataset.from_pandas(test_df)
    })

@task
def tokenize_data(dataset, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='longest')
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer

@task
def train_model(train_dataset, eval_dataset, model_checkpoint, training_args):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        accuracy = accuracy_score(labels, predictions)
        mlflow.log_metric("batch_accuracy", accuracy)
        return {'accuracy': accuracy}

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_checkpoint))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer

@task
def evaluate_model(trainer, dataset, dataset_name="test"):
    """Evaluates the model and logs failure cases for training data improvement"""
    predictions_output = trainer.predict(dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = dataset['label']
    texts = dataset['text']

    # Basic metrics for MLflow
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    mlflow.log_metrics({
        f"{dataset_name}_accuracy": acc,
        f"{dataset_name}_f1": f1,
    })

    # Analyze failure cases
    errors_df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predictions,
        'correct': true_labels == predictions
    })
    
    # Save failure analysis
    errors_df[errors_df['correct'] == False].to_csv(
        Path(__file__).resolve().parent / "results" / f"{dataset_name}_errors.csv",
        index=False
    )

    logger.info(f"{dataset_name} set - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1

@task
def save_model_and_results(trainer, tokenizer, model_dir):
    """Save just what's needed for inference"""
    # Save model and tokenizer to the location FastAPI expects
    model_save_path = model_dir / "saved_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Log basic metrics to MLflow for portfolio
    mlflow.log_artifact(str(model_save_path))

def setup_mlflow():
    """Configure MLflow tracking"""
    experiment_dir = Path(__file__).resolve().parent / "mlruns"
    mlflow.set_tracking_uri(f"file://{experiment_dir.absolute()}")
    
    # Set or create the experiment
    experiment = mlflow.set_experiment("bert_training")
    
    # End any existing runs that might be lingering
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

@flow(log_prints=False)
def training_flow():
    """Main training flow that orchestrates the entire process."""
    with mlflow.start_run() as run:
        # Initialize
        model_dir = setup_directories()
        logger.info("Starting model training")
        
        # Model configuration
        model_checkpoint = "answerdotai/ModernBERT-base"
        training_args = TrainingArguments(
            output_dir='checkpoints',
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # Reduce noise in output
            disable_tqdm=True,
            report_to=["mlflow"],
            logging_dir=None
        )
        
        # Basic parameter logging for MLflow
        mlflow.log_params({
            "model_checkpoint": model_checkpoint,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
        })
        
        # Training pipeline
        df = create_dataframe()
        train_df, test_df = split_data(df)
        dataset_dict = create_hf_datasets(train_df, test_df)
        tokenized_dataset, tokenizer = tokenize_data(dataset_dict, model_checkpoint)
        trainer = train_model(tokenized_dataset["train"], tokenized_dataset["test"], 
                            model_checkpoint, training_args)
        
        # Evaluate and save error cases for analysis
        evaluate_model(trainer, tokenized_dataset["train"], "train")
        evaluate_model(trainer, tokenized_dataset["test"], "test")
        
        # Save model for FastAPI
        save_model_and_results(trainer, tokenizer, model_dir)
        
        logger.info("Training completed - model saved to saved_model/")
        return trainer, tokenizer

if __name__ == "__main__":
    training_flow()