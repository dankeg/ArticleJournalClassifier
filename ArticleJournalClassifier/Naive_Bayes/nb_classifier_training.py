import time
from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from ArticleJournalClassifier.Data.data_split import load_data, split_data

DATA_USAGE_FRACTION = 1
TRAIN_SPLIT_FRACTION = 0.8
MODEL_PATH = Path("nb_data/trained_nb_model.joblib")


def train_and_save_model(
    train_texts,
    train_labels,
    model_path: Path = MODEL_PATH,
    ngram_range: tuple = (2, 3),
):
    """
    Builds a Naive Bayes pipeline, trains it, and saves it to disk.

    Parameters:
        train_texts (list): Training text data.
        train_labels (list): Training labels.
        model_path (Path): File path to save the trained model.
        ngram_range (tuple): N-gram range for the CountVectorizer.

    Returns:
        pipe: The trained pipeline model.
        training_time (float): The duration (in seconds) taken to train the model.
    """
    pipe = make_pipeline(CountVectorizer(ngram_range=ngram_range), MultinomialNB())

    print("Starting Training")
    start_time = time.time()
    pipe.fit(train_texts, train_labels)
    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")

    dump(pipe, model_path)
    print(f"Model saved at {model_path.resolve()}")

    return pipe, training_time


if __name__ == "__main__":
    data_file = "top_journals_text.json"
    loaded_data = load_data(data_file)

    train_texts, train_labels, eval_texts, eval_labels, index_mapping = split_data(
        loaded_data, DATA_USAGE_FRACTION, TRAIN_SPLIT_FRACTION
    )

    model, training_time = train_and_save_model(train_texts, train_labels)
