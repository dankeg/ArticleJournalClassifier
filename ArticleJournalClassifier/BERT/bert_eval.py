from typing import Dict

import nltk
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
)

from ArticleJournalClassifier.Data.data_split import load_and_prepare_data

nltk.download("punkt")

# Constants
JSON_FILE_PATH: str = "top_journals_text.json"
MODEL_DIR: str = "bert-finetuned-top-journals"
CHUNK_SIZE: int = 128
DATA_USAGE_FRACTION: float = 1
TRAIN_SPLIT_FRACTION: float = 0.8
SEED: int = 42
MODEL_NAME: str = "bert-base-uncased"


def evaluate_model(
    encoded: Dict,
    tokenizer,
    model_dir: str,
) -> None:
    """
    Loads the saved model and evaluates it on the test set using the provided dataset and tokenizer.

    Args:
        encoded (Dict): The pre-processed dataset containing training and validation splits.
        tokenizer: The tokenizer used for processing the data.
        model_dir (str): Directory where the trained model is saved.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted"),
            "precision": precision_score(p.label_ids, preds, average="weighted"),
            "recall": recall_score(p.label_ids, preds, average="weighted"),
            "report": classification_report(p.label_ids, preds),
        }

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        eval_dataset=encoded["validation"],
    )
    results = trainer.evaluate()
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    encoded_dataset, tokenizer, label2id, id2label = load_and_prepare_data(
        JSON_FILE_PATH,
        CHUNK_SIZE,
        DATA_USAGE_FRACTION,
        TRAIN_SPLIT_FRACTION,
        SEED,
        MODEL_NAME,
    )

    evaluate_model(
        encoded=encoded_dataset,
        tokenizer=tokenizer,
        model_dir=MODEL_DIR,
    )
