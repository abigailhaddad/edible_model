import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,

)
from scipy.special import softmax
from sklearn.metrics import f1_score, classification_report

def setup_directories():
    """Ensure required directories exist."""
    model_dir = Path(__file__).resolve().parent
    paths = [model_dir / "synthetic_data", model_dir / "results", model_dir / "checkpoints"]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def create_dataframe():
    """
    Loads the JSON file containing labeled data and returns a pandas DataFrame.
    """
    model_dir = Path(__file__).resolve().parent
    src_dir = model_dir.parent
    data_path = src_dir / "model" / "synthetic_data" / "data.json"

    
    with open(data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["labeled_data"])
    return df

def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def create_hf_datasets(train_df, test_df):
    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "test": datasets.Dataset.from_pandas(test_df)
    })

def tokenize_data(dataset, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='longest')
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer

def train_model(train_dataset, eval_dataset, model_checkpoint, training_args):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return {'accuracy': accuracy_score(labels, predictions)}

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

def evaluate_model(trainer, dataset, dataset_name="test"):
    """
    Evaluates the model using the dataset and logs key metrics, including the confusion matrix breakdown
    and actual examples for TP, FP, TN, and FN.
    """
    predictions_output = trainer.predict(dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = dataset['label']
    texts = dataset['text']  # Extract text examples

    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Extracting TP, FP, TN, FN explicitly
    tn, fp, fn, tp = cm.ravel()

    # Identify examples in each category
    tp_examples = [(texts[i], true_labels[i], predictions[i]) for i in range(len(predictions)) if true_labels[i] == 1 and predictions[i] == 1]
    fp_examples = [(texts[i], true_labels[i], predictions[i]) for i in range(len(predictions)) if true_labels[i] == 0 and predictions[i] == 1]
    tn_examples = [(texts[i], true_labels[i], predictions[i]) for i in range(len(predictions)) if true_labels[i] == 0 and predictions[i] == 0]
    fn_examples = [(texts[i], true_labels[i], predictions[i]) for i in range(len(predictions)) if true_labels[i] == 1 and predictions[i] == 0]

    logger.info(f"Evaluation on {dataset_name} set:")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Classification Report:\n{classification_report(true_labels, predictions)}")

    # Log the counts of each category
    logger.info(f"Breakdown:\n"
                f"True Positives (TP): {tp} \n"
                f"False Positives (FP): {fp} \n"
                f"True Negatives (TN): {tn} \n"
                f"False Negatives (FN): {fn} \n")

    # Log actual examples
    def log_examples(category, examples, limit=5):
        """Helper function to log example texts with true and predicted labels"""
        logger.info(f"\n{category} Examples ({len(examples)} found, showing up to {limit}):")
        for text, true_label, predicted_label in examples[:limit]:
            logger.info(f"Text: {text}\n  True Label: {true_label}, Predicted: {predicted_label}\n")

    log_examples("True Positives (TP)", tp_examples)
    log_examples("False Positives (FP)", fp_examples)
    log_examples("True Negatives (TN)", tn_examples)
    log_examples("False Negatives (FN)", fn_examples)

    return cm, acc, f1, tp, fp, tn, fn


def generate_predictions_dataframe(trainer, tokenized_dataset):
    def get_predictions(dataset, dataset_name):
        texts = [item['text'] for item in dataset]
        true_labels = [item['label'] for item in dataset]
        predictions_output = trainer.predict(dataset)
        logits = predictions_output.predictions
        predictions = np.argmax(logits, axis=-1)
        probabilities = softmax(logits, axis=1)

        return pd.DataFrame({
            'text': texts,
            'test_or_train': [dataset_name] * len(texts),
            'real_label': true_labels,
            'model_label': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1],
        })

    train_df = get_predictions(tokenized_dataset["train"], 'train')
    test_df = get_predictions(tokenized_dataset["test"], 'test')

    return pd.concat([train_df, test_df], ignore_index=True)

def gen_model(df, model_checkpoint, training_args):
    train_df, test_df = split_data(df)
    dataset_dict = create_hf_datasets(train_df, test_df)
    tokenized_dataset, tokenizer = tokenize_data(dataset_dict, model_checkpoint)
    trainer = train_model(tokenized_dataset["train"], tokenized_dataset["test"], model_checkpoint, training_args)
    
    evaluate_model(trainer, tokenized_dataset["train"])
    evaluate_model(trainer, tokenized_dataset["test"])
    
    model_dir = Path(__file__).resolve().parent
    results_csv_path = model_dir / "results" / "model_predictions.csv"
    results_df = generate_predictions_dataframe(trainer, tokenized_dataset)
    results_df.to_csv(results_csv_path, index=False)

    model_dir = Path(__file__).resolve().parent
    trainer.save_model(model_dir / "saved_model")
    tokenizer.save_pretrained(model_dir / "saved_model")
    
    return trainer, tokenizer

model_dir = Path(__file__).resolve().parent
log_file_path = model_dir / "results" / "training.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_directories()
    logger.info("Starting model training")
    
    df = create_dataframe()
    logger.info(f"Created dataset with {len(df)} examples")
    
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
        logging_steps=10,
    )
    
    trainer, tokenizer = gen_model(df, model_checkpoint, training_args)
    logger.info("Model training completed")