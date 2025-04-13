from pathlib import Path

from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ArticleJournalClassifier.Data.data_split import load_data, split_data

DATA_USAGE_FRACTION = 1
TRAIN_SPLIT_FRACTION = 0.8
MODEL_PATH = Path("nb_data/trained_nb_model.joblib")


def evaluate_model(eval_texts, eval_labels, model_path: Path = MODEL_PATH):
    """
    Loads the Naive Bayes model and measures performance on the evaluation/test data.

    Parameters:
        eval_texts (list): Evaluation text data.
        eval_labels (list): Evaluation labels.
        model_path (Path): File path of the saved model.

    Returns:
        metrics (dict): Accuracy, precision, recall, and F1 scores.
        report (str): Classification report.
    """
    reloaded_pipe = load(model_path)
    y_pred = reloaded_pipe.predict(eval_texts)

    accuracy = accuracy_score(eval_labels, y_pred)
    precision = precision_score(eval_labels, y_pred, average="weighted")
    recall = recall_score(eval_labels, y_pred, average="weighted")
    f1 = f1_score(eval_labels, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

    report = classification_report(eval_labels, y_pred)
    print(report)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics, report


if __name__ == "__main__":
    data_file = "top_journals_text.json"
    loaded_data = load_data(data_file)

    train_texts, train_labels, eval_texts, eval_labels, index_mapping = split_data(
        loaded_data, DATA_USAGE_FRACTION, TRAIN_SPLIT_FRACTION
    )

    metrics, classification_report_str = evaluate_model(eval_texts, eval_labels)
