import json
import random

import nltk
import numpy as np
import torch

from ArticleJournalClassifier.RNN.rnn_train import (
    LSTMClassifier,
    build_embedding_matrix,
    evaluate,
    prepare_data_pipeline,
)

from gensim.models import Word2Vec

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS")
else:
    device = torch.device("cpu")
    print("Using device: CPU")


JSON_FILE_PATH: str = "top_journals_text.json"
BATCH_SIZE: int = 512
CHUNK_SIZE: int = 500
WORD2VEC_DIM: int = 100
WINDOW_SIZE: int = 5
MIN_COUNT: int = 1
SEED: int = 42

HIDDEN_DIM: int = 128

MODEL_SAVE_PATH: str = "rnn_data/rnn.pt"

DATA_USAGE_FRACTION: float = 1
TRAIN_SPLIT_FRACTION: float = 0.8

# Set seeds for reproducibility and download nltk's tokenizer
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
nltk.download("punkt")


def evaluate_model(vocab, test_dataset, train_tokens, label2id) -> None:
    num_classes = len(label2id)

    # w2v_model = Word2Vec(
    #     train_tokens,
    #     vector_size=WORD2VEC_DIM,
    #     window=WINDOW_SIZE,
    #     min_count=MIN_COUNT,
    #     workers=4,
    # )

    
    # embedding_matrix = build_embedding_matrix(vocab, w2v_model)

    state = torch.load(MODEL_SAVE_PATH, map_location=device)
    emb_key = "embedding.weight"
    fc_key  = "fc.weight"
    
    emb_rows, emb_dim = state[emb_key].shape

    dummy_matrix = torch.empty(emb_rows, emb_dim)

    model = LSTMClassifier(dummy_matrix, HIDDEN_DIM, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Model loaded from {MODEL_SAVE_PATH}")

    f1, precision, accuracy, recall, report = evaluate(
        model, test_dataset, device, batch_size=BATCH_SIZE
    )

    print(f"F1 Score: {f1}")
    print(f"Precision Score: {precision}")
    print(f"Accuracy Score: {accuracy}")
    print(f"Recall Score: {recall}")
    print(report)


def main():
    VOCAB_SAVE_PATH = "rnn_data/vocab.json"
    INDEX2VOCAB_SAVE_PATH = "rnn_data/index2vocab.json"
    
    with open(VOCAB_SAVE_PATH, "r") as f:
        vocab = json.load(f)
    
    with open(INDEX2VOCAB_SAVE_PATH, "r") as f:
        index2vocab = json.load(f)

    vocab, _, test_dataset, train_tokens, label2id = prepare_data_pipeline(
        JSON_FILE_PATH, data_usage_fraction=DATA_USAGE_FRACTION, train_split_fraction=TRAIN_SPLIT_FRACTION, vocab=vocab
    )

    evaluate_model(vocab, test_dataset, train_tokens, label2id)


if __name__ == "__main__":
    main()
