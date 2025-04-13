# reusable_data.py
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize
from transformers import (
    AutoTokenizer,
)


def load_data(file_path: str) -> Any:
    """
    Load JSON data given a file path

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def split_data(
    data_dict: Dict[str, List[str]],
    data_usage_fraction: float,
    train_split_fraction: float,
    seed: int = 42,
) -> Tuple:
    """
    Split articles into training and validation sets for each class.

    For each class in the dictionary, the function shuffles the articles,
    retains a fraction of them, and splits the result into training and validation lists.

    Args:
        data_dict (Dict[str, List[str]]): Class names mapped to articles
        data_usage_fraction (float): Fraction of articles per class to use.
        train_split_fraction (float): Fraction of remaining articles for training, the rest being test.
        seed (int): Seed, making data split consistent across iterations.

    Returns:
        Tuple: Train and Test data & labels, as well as vocab mapping
    """
    random.seed(seed)
    np.random.seed(seed)

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    index_dict = {}
    current_index = 0

    for journal_class, articles in data_dict.items():
        if journal_class not in index_dict:
            index_dict[journal_class] = current_index
            current_index += 1

        random.shuffle(articles)
        n_keep = int(len(articles) * data_usage_fraction)
        selected = articles[:n_keep]
        split_idx = int(len(selected) * train_split_fraction)
        train_split, val_split = selected[:split_idx], selected[split_idx:]

        train_texts.extend(train_split)
        train_labels.extend([index_dict[journal_class]] * len(train_split))
        val_texts.extend(val_split)
        val_labels.extend([index_dict[journal_class]] * len(val_split))

    return train_texts, train_labels, val_texts, val_labels, index_dict


def chunk_tokens(tokens: List[str], chunk_size: int) -> List[List[str]]:
    """
    Break articles into easier to process chunks

    Args:
        tokens (List[str]): List of tokens.
        chunk_size (int): Maximum number of tokens per chunk.

    Returns:
        List[List[str]]: List of token chunks.
    """
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]


def preprocess_articles(
    texts: List[str], labels: List[int], chunk_size: int
) -> Tuple[List[str], List[int]]:
    """
    Tokenize texts, chunk the tokens, and reconstruct text chunks.

    Each original article is tokenized and split into chunks that are joined back into strings.
    The corresponding label is duplicated for each chunk.

    Args:
        texts (List[str]): List of article texts.
        labels (List[int]): List of labels corresponding to each text.
        chunk_size (int): Maximum number of tokens per chunk.

    Returns:
        Tuple: Chunked training data, with corresponding labels
    """
    new_texts, new_labels = [], []
    for text, label in zip(texts, labels):
        tokens = word_tokenize(text)
        chunks = chunk_tokens(tokens, chunk_size)
        new_texts.extend([" ".join(chunk) for chunk in chunks])
        new_labels.extend([label] * len(chunks))
    return new_texts, new_labels


def load_and_prepare_data(
    json_path: str,
    chunk_size: int = 128,
    data_usage_fraction: float = 1,
    train_split_fraction: float = 0.8,
    seed: int = 42,
    model_name: str = "bert-base-uncased",
) -> Tuple:
    """
    Load, split, preprocess, and tokenize a JSON dataset for BERT.

    Args:
        json_path: Path to the JSON file.
        chunk_size: Maximum tokens per chunk.
        data_usage_fraction: Fraction of articles to use per class.
        train_split_fraction: Fraction of articles for training.
        seed: Random seed.
        model_name: Pre-trained model used for tokenization.

    Returns:
        Tuple with data needed to perform training and eval
    """
    data_dict = load_data(json_path)
    train_texts, train_labels, val_texts, val_labels, index_dict = split_data(
        data_dict, data_usage_fraction, train_split_fraction, seed
    )
    train_texts, train_labels = preprocess_articles(
        train_texts, train_labels, chunk_size
    )
    val_texts, val_labels = preprocess_articles(val_texts, val_labels, chunk_size)

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=chunk_size,
        )

    encoded = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    label2id = index_dict
    id2label = {v: k for k, v in label2id.items()}

    return encoded, tokenizer, label2id, id2label
