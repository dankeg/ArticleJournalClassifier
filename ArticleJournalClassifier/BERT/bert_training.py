from typing import Any, Dict

import matplotlib.pyplot as plt
import nltk
import numpy as np
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ArticleJournalClassifier.Data.data_split import load_and_prepare_data

nltk.download("punkt")

# Constants
JSON_FILE_PATH: str = "top_journals_text.json"
MODEL_DIR: str = "bert-finetuned-top-journals"
CHUNK_SIZE: int = 128
DATA_USAGE_FRACTION: float = 1.0
TRAIN_SPLIT_FRACTION: float = 0.8
SEED: int = 42
MODEL_NAME: str = "bert-base-uncased"


def train_model(
    encoded: DatasetDict,
    tokenizer: AutoTokenizer,
    label2id: Dict[Any, Any],
    id2label: Dict[Any, Any],
    output_dir: str = "bert-finetuned-top-journals",
    num_epochs: int = 10,
    per_device_train_batch: int = 64,
    per_device_eval_batch: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    model_name: str = "bert-base-uncased",
) -> Trainer:
    """
    Fine-tunes a BERT model for sequence classification.

    Args:
        encoded (DatasetDict): Tokenized datasets with train and validation splits.
        tokenizer (AutoTokenizer): Pre-trained tokenizer.
        label2id (Dict[Any, Any]): Mapping from label names to IDs.
        id2label (Dict[Any, Any]): Reverse mapping from IDs to label names.
        output_dir (str): Directory for saving model checkpoints.
        num_epochs (int): Number of training epochs.
        per_device_train_batch (int): Training batch size per device.
        per_device_eval_batch (int): Evaluation batch size per device.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        seed (int): Random seed.
        model_name (str): Identifier of the pre-trained model.

    Returns:
        Trainer: Trainer instance with the fine-tuned model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """
        Computes evaluation metrics.

        Args:
            p (EvalPrediction): Predictions and true labels.

        Returns:
            Dict[str, float]: Dictionary with accuracy, f1, precision, and recall.
        """
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted"),
            "precision": precision_score(p.label_ids, preds, average="weighted"),
            "recall": recall_score(p.label_ids, preds, average="weighted"),
        }

    class MetricsPlottingCallback(TrainerCallback):
        """
        Callback to collect and plot training and evaluation metrics.
        """

        def __init__(self) -> None:
            self.train_metrics: Dict[str, list[tuple]] = {}
            self.eval_metrics: Dict[str, list[tuple]] = {}

        def on_log(
            self, args, state, control, logs: Dict[str, Any] = None, **kwargs
        ) -> Any:
            """
            Collects metrics from logs during training.

            Args:
                args: Training arguments.
                state: Current training state.
                control: Trainer control object.
                logs (Dict[str, Any], optional): Metrics logged in the current step.
            """
            if logs:
                if any(key.startswith("eval_") for key in logs.keys()):
                    for key, value in logs.items():
                        self.eval_metrics.setdefault(key, []).append(
                            (state.global_step, value)
                        )
                else:
                    for key, value in logs.items():
                        self.train_metrics.setdefault(key, []).append(
                            (state.global_step, value)
                        )
            return control

        def on_train_end(self, args, state, control, **kwargs) -> Any:
            """
            Plots and saves the collected metrics at the end of training.

            Args:
                args: Training arguments.
                state: Final training state.
                control: Trainer control object.
            """
            for metric, values in self.train_metrics.items():
                steps, metric_values = zip(*values)
                plt.figure()
                plt.plot(steps, metric_values, marker="o")
                plt.title(f"Training {metric} over Steps")
                plt.xlabel("Global Step")
                plt.ylabel(metric)
                plt.grid(True)
                plt.savefig(f"training_{metric}.png")
                plt.show()
            for metric, values in self.eval_metrics.items():
                steps, metric_values = zip(*values)
                plt.figure()
                plt.plot(steps, metric_values, marker="o", linestyle="--")
                plt.title(f"Evaluation {metric} over Steps")
                plt.xlabel("Global Step")
                plt.ylabel(metric)
                plt.grid(True)
                plt.savefig(f"evaluation_{metric}.png")
                plt.show()
            return control

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch,
        per_device_eval_batch_size=per_device_eval_batch,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MetricsPlottingCallback()],
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(output_dir)
    return trainer


if __name__ == "__main__":
    # Load data, train the model, and save it to disk.
    encoded_dataset, tokenizer, label2id, id2label = load_and_prepare_data(
        JSON_FILE_PATH,
        CHUNK_SIZE,
        DATA_USAGE_FRACTION,
        TRAIN_SPLIT_FRACTION,
        SEED,
        MODEL_NAME,
    )
    train_model(
        encoded_dataset,
        tokenizer,
        label2id,
        id2label,
        output_dir=MODEL_DIR,
        num_epochs=10,
        model_name=MODEL_NAME,
        seed=SEED,
    )
