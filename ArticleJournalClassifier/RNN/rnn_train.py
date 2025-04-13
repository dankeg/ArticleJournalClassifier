import gc
import json
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from ArticleJournalClassifier.Data.data_split import chunk_tokens, load_data, split_data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
    print("Using device: CUDA")
elif torch.backends.mps.is_available():
    device: torch.device = torch.device("mps")
    print("Using device: MPS")
else:
    device: torch.device = torch.device("cpu")
    print("Using device: CPU")


JSON_FILE_PATH: str = "top_journals_text.json"
BATCH_SIZE: int = 512
CHUNK_SIZE: int = 500
WORD2VEC_DIM: int = 100
WINDOW_SIZE: int = 5
MIN_COUNT: int = 1
SEED: int = 42

HIDDEN_DIM: int = 128
NUM_EPOCHS: int = 20
LEARNING_RATE: float = 0.0001

MODEL_SAVE_PATH: str = "rnn_data/rnn.pt"

DATA_USAGE_FRACTION: float = 1
TRAIN_SPLIT_FRACTION: float = 0.8

# Set seeds for reproducibility and download nltk's tokenizer
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
nltk.download("punkt")


def prepare_data_pipeline(
    json_path: str,
    data_usage_fraction: float = DATA_USAGE_FRACTION,
    train_split_fraction: float = TRAIN_SPLIT_FRACTION,
    vocab: dict = None
) -> Tuple:
    """Prepare data pipeline from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        data_usage_fraction (float): Fraction of data to use.
        train_split_fraction (float): Fraction of training data.

    Returns:
        Tuple: Vocabulary, train dataset, test dataset, train tokens, and label2id mapping.
    """
    data_dict = load_data(json_path)
    train_texts, train_labels, test_texts, test_labels, label2id = split_data(
        data_dict, data_usage_fraction, train_split_fraction, seed=SEED
    )

    train_tokens: List[List[str]] = []
    train_token_labels: List[Any] = []
    for text, label in zip(train_texts, train_labels):
        tokens = word_tokenize(text)
        chunks = chunk_tokens(tokens, CHUNK_SIZE)
        train_tokens.extend(chunks)
        train_token_labels.extend([label] * len(chunks))

    test_tokens: List[List[str]] = []
    test_token_labels: List[Any] = []
    for text, label in zip(test_texts, test_labels):
        tokens = word_tokenize(text)
        chunks = chunk_tokens(tokens, CHUNK_SIZE)
        test_tokens.extend(chunks)
        test_token_labels.extend([label] * len(chunks))

    # Build the vocabulary from training tokens
    if vocab is None:
        vocab: Dict[str, int] = build_vocab(train_tokens)

    # Vectorize both train and test datasets
    train_dataset = vectorize_dataset(train_tokens, train_token_labels, vocab)
    test_dataset = vectorize_dataset(test_tokens, test_token_labels, vocab)

    return vocab, train_dataset, test_dataset, train_tokens, label2id


def build_vocab(token_sequences: List[List[str]]) -> Dict[str, int]:
    """Build a custom vocabulary mapping tokens to indices.

    Args:
        token_sequences (List[List[str]]): List of token sequences.

    Returns:
        Dict[str, int]: A dictionary mapping tokens to indices.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tokens in token_sequences:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def vectorize_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """Convert a list of tokens into a list of indices.

    Args:
        tokens (List[str]): A list of tokens.
        vocab (Dict[str, int]): Mapping of tokens to indices.

    Returns:
        List[int]: Indices corresponding to tokens.
    """
    unk_idx = vocab["<UNK>"]
    return [vocab.get(token, unk_idx) for token in tokens]


def vectorize_dataset(
    token_sequences: List[List[str]], labels: List[Any], vocab: Dict[str, int]
) -> List[Tuple[List[int], Any]]:
    """Create a dataset of vectorized token sequences and labels.

    Args:
        token_sequences (List[List[str]]): List of token sequences.
        labels (List[Any]): List of labels.
        vocab (Dict[str, int]): Vocabulary mapping tokens to indices.

    Returns:
        List[Tuple[List[int], Any]]: A list of (vectorized tokens, label) pairs.
    """
    dataset = []
    for tokens, label in zip(token_sequences, labels):
        indices = vectorize_tokens(tokens, vocab)
        dataset.append((indices, label))
    return dataset


def build_embedding_matrix(vocab: Dict[str, int], w2v_model: Word2Vec) -> torch.Tensor:
    """Creates an embedding matrix, given pretrained embeddings.

    This function is somewhat redundant, and was used heavily when testing other
    pretrained embedding schemes, such as Glove. Such embeddings lack much of the
    specialized academic paper vocab, requiring this scheme to include and fine tune this vocab.
    Due to the poor results using this, Word2Vec was used instead: which has a 100% hit rate given
    that it is initialized using the train vocab to begin with.

    Thus, this is included to enable testing in the future.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping tokens to indices.
        w2v_model (Word2Vec): A trained Word2Vec model.

    Returns:
        torch.Tensor: An embedding matrix.
    """
    vocab_size = len(vocab)
    embedding_matrix = torch.empty(vocab_size, WORD2VEC_DIM)
    pad_idx = vocab["<PAD>"]
    unk_idx = vocab["<UNK>"]
    embedding_matrix[pad_idx] = torch.zeros(WORD2VEC_DIM)
    embedding_matrix[unk_idx] = torch.empty(WORD2VEC_DIM).uniform_(-0.05, 0.05)

    hits, misses = 0, 0
    for token, idx in vocab.items():
        if token in ("<PAD>", "<UNK>"):
            continue
        if token in w2v_model.wv:
            embedding_matrix[idx] = torch.tensor(w2v_model.wv[token])
            hits += 1
        else:
            embedding_matrix[idx] = torch.empty(WORD2VEC_DIM).uniform_(-0.05, 0.05)
            misses += 1
    print(
        f"Word2Vec coverage: {hits}/{hits + misses} tokens ({100 * hits / (hits + misses):.2f}%)"
    )
    return embedding_matrix


class LSTMClassifier(nn.Module):
    """Simple LSTM-based classifier model."""

    def __init__(
        self, embedding_matrix: torch.Tensor, hidden_dim: int, num_classes: int
    ) -> None:
        """Initialize LSTMClassifier.

        Args:
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_classes (int): Number of target classes.
        """
        super(LSTMClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the model.

        Args:
            x (torch.Tensor): Input chunk, as tokens mapped to indices.

        Returns:
            torch.Tensor: Output logits tensor.
        """
        embeds = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embeds)
        final_hidden = h_n[-1]
        logits = self.fc(final_hidden)
        return logits


def train_epoch(
    model: nn.Module,
    dataset: List[Tuple[List[int], Any]],
    optimizer: Any,
    loss_fn: Any,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Train the model for a single epoch, generating relevant information.

    Args:
        model (nn.Module): Model to train.
        dataset (List[Tuple[List[int], Any]]): Dataset containing (token indices, label) pairs.
        optimizer (Any): Optimizer for training.
        loss_fn (Any): Loss function.
        device (torch.device): Device to run the model on.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Training"):
        batch = dataset[i : i + batch_size]
        sequences = [torch.tensor(seq, dtype=torch.long) for seq, _ in batch]
        labels = [torch.tensor(lbl, dtype=torch.long) for _, lbl in batch]

        X = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0).to(
            device
        )
        y = torch.stack(labels).to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        n_samples += len(batch)
        del X, y, preds, loss, batch, sequences, labels
        gc.collect()

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    dataset: List[Tuple[List[int], Any]],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> Tuple:
    """Evaluate the model and compute relevant metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataset (List[Tuple[List[int], Any]]): Dataset containing (token indices, label) pairs.
        device (torch.device): Device to run the evaluation on.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.

    Returns:
        Tuple: A tuple containing F1, precision, accuracy, recall, and a classification report.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[Any] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i : i + batch_size]
            sequences = [torch.tensor(seq, dtype=torch.long) for seq, _ in batch]
            labels = [lbl for _, lbl in batch]
            X = nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=0
            ).to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Clean up memory
            del X, outputs, batch, sequences, labels
            gc.collect()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds)

    return f1, precision, accuracy, recall, report


def plot_training(
    epochs_list: List[int],
    losses_list: List[float],
    accuracies_list: List[float],
    f1_list: List[float],
    precision_list: List[float],
    recall_list: List[float],
) -> None:
    """Generate and save loss and evaluation metrics following training.

    Args:
        epochs_list (List[int]): List of epoch numbers.
        losses_list (List[float]): List of loss values.
        accuracies_list (List[float]): List of accuracy values.
        f1_list (List[float]): List of F1 scores.
        precision_list (List[float]): List of precision values.
        recall_list (List[float]): List of recall values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, losses_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("rnn_loss_training.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, losses_list, label="Loss")
    plt.plot(epochs_list, accuracies_list, label="Accuracy")
    plt.plot(epochs_list, f1_list, label="F1 Score")
    plt.plot(epochs_list, precision_list, label="Precision")
    plt.plot(epochs_list, recall_list, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("rnn_metrics_training.png")
    plt.close()


def train_model(
    vocab: Dict[str, int],
    train_dataset: List[Tuple[List[int], Any]],
    train_tokens: List[List[str]],
    label2id: Dict[Any, int],
) -> None:
    """Train the model and save training plots and model weights.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping tokens to indices.
        train_dataset (List[Tuple[List[int], Any]]): Training dataset.
        train_tokens (List[List[str]]): List of tokenized training texts.
        label2id (Dict[Any, int]): Label to ID mapping.
    """
    num_classes = len(label2id)

    w2v_model = Word2Vec(
        train_tokens,
        vector_size=WORD2VEC_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=4,
    )
    embedding_matrix = build_embedding_matrix(vocab, w2v_model)

    model = LSTMClassifier(embedding_matrix, HIDDEN_DIM, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs_list: List[int] = []
    losses_list: List[float] = []
    accuracies_list: List[float] = []
    f1_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_epoch(
            model, train_dataset, optimizer, loss_fn, device, batch_size=BATCH_SIZE
        )
        f1, precision, accuracy, recall, report = evaluate(
            model, train_dataset, device, batch_size=BATCH_SIZE
        )
        epochs_list.append(epoch + 1)
        losses_list.append(epoch_loss)
        accuracies_list.append(accuracy)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.3f}")
        print(report)

    plot_training(
        epochs_list, losses_list, accuracies_list, f1_list, precision_list, recall_list
    )
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    VOCAB_SAVE_PATH = "rnn_data/vocab2index.json"
    with open(VOCAB_SAVE_PATH, "w") as f:
        json.dump(vocab, f)

    index2vocab = {idx: token for token, idx in vocab.items()}

    INDEX2VOCAB_SAVE_PATH = "rnn_data/index2vocab.json"
    with open(INDEX2VOCAB_SAVE_PATH, "w") as f:
        json.dump(index2vocab, f)


def main() -> None:
    """Run the data preparation and model training pipeline."""
    vocab, train_dataset, _, train_tokens, label2id = prepare_data_pipeline(
        JSON_FILE_PATH,
        data_usage_fraction=DATA_USAGE_FRACTION,
        train_split_fraction=TRAIN_SPLIT_FRACTION,
    )
    train_model(vocab, train_dataset, train_tokens, label2id)


if __name__ == "__main__":
    main()
