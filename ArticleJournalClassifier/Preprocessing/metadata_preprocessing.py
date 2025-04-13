import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process


def extract_base_journal(raw_journal: str) -> str:
    """
    Simplify a raw journal reference.

    Args:
        raw_journal (str): The original journal reference string.

    Returns:
        str: A cleaned and simplified journal name.
    """
    if not raw_journal:
        return ""
    text = raw_journal.lower()

    # Regular expression removes dates, issue numbers, years, bracketed information, etc.
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\d+([\-–]\d+)?", "", text)
    text = re.sub(r"[^a-z.\s]", "", text)
    text = re.sub(r"\.+", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_match_base_journal(
    base_journal: str,
    global_known_stems: List[str],
    global_stem_map: Dict[str, str],
    threshold: int = 90,
) -> str:
    """
    Fuzzy-match a journal to a canonical name.

    Args:
        base_journal (str): Simplified journal string.
        global_known_stems (List[str]): List of known journal stems.
        global_stem_map (Dict[str, str]): Mapping of stems to canonical names.
        threshold (int): Matching score threshold.

    Returns:
        str: The canonical journal name.
    """
    if not base_journal:
        return ""
    if global_known_stems:
        best_match, score, _ = rf_process.extractOne(
            base_journal, global_known_stems, scorer=rf_fuzz.token_sort_ratio
        )
        if score >= threshold:
            return global_stem_map[best_match]
    global_known_stems.append(base_journal)
    global_stem_map[base_journal] = base_journal
    return base_journal


def plot_top_journals(
    sorted_items: List[Tuple[str, int]], top_n: int, output_path: str
) -> None:
    """
    Plot and save a pie chart of the top journals.

    Args:
        sorted_items (List[Tuple[str, int]]): List of (journal, count) tuples.
        top_n (int): Number of top journals to plot.
        output_path (str): File path for saving the chart.
    """
    labels = [f"{name} ({count})" for name, count in sorted_items[:top_n]]
    values = [count for _, count in sorted_items[:top_n]]

    plt.figure()
    plt.pie(values, labels=labels, startangle=90)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved pie chart to {output_path}.")


def process_journals(
    file_path: str,
    threshold: int = 90,
    top_n: int = 20,
    output_path: str = "pie_chart.png",
    max_records: int = None,
) -> Tuple[List[Tuple[str, int]], Dict[str, List[Any]]]:
    """
    Process a JSON lines file of journal records and plot the top journals.

    Reads each record, cleans and matches the journal name,
    counts occurrences, and plots a pie chart of the top journals.

    Args:
        file_path (str): Path to the JSON lines file.
        threshold (int): Fuzzy matching threshold.
        top_n (int): Number of journals to include in the chart.
        output_path (str): File path to save the chart.
        max_records (int, optional): Maximum number of records to process.

    Returns:
        Tuple[List[Tuple[str, int]], Dict[str, List[Any]]]: A sorted list of journal counts and a mapping of journals to arXiv IDs.
    """
    global_counter = Counter()
    global_j2ids: Dict[str, List[Any]] = {}
    global_known_stems: List[str] = []
    global_stem_map: Dict[str, str] = {}
    processed_so_far = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records and processed_so_far >= max_records:
                break
            processed_so_far += 1
            data = json.loads(line)
            raw_journal = data.get("journal-ref", "")
            arxiv_id = data.get("id", None)
            base_stem = extract_base_journal(raw_journal)
            if not base_stem:
                continue

            canonical_name = fuzzy_match_base_journal(
                base_journal=base_stem,
                global_known_stems=global_known_stems,
                global_stem_map=global_stem_map,
                threshold=threshold,
            )
            global_counter[canonical_name] += 1
            if arxiv_id:
                if canonical_name not in global_j2ids:
                    global_j2ids[canonical_name] = []
                global_j2ids[canonical_name].append(arxiv_id)

            if processed_so_far % 10000 == 0:
                print(f"Processed {processed_so_far} lines.")

    sorted_items = sorted(global_counter.items(), key=lambda x: x[1], reverse=True)
    plot_top_journals(sorted_items, top_n, output_path)

    return sorted_items, global_j2ids


def save_journal_id_map(
    journal_to_ids: Dict[str, List[Any]], output_path: str = "journal_id_map.json"
) -> None:
    """
    Save journal-to-arXiv ID mapping to a JSON file.

    Args:
        journal_to_ids (Dict[str, List[Any]]): Mapping of journals to arXiv IDs.
        output_path (str): File path for the JSON output.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(journal_to_ids, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
    sorted_journals, journal_id_map = process_journals(
        file_path=INPUT_FILE,
        threshold=85,
        top_n=20,
        output_path="pie_chart.png",
        max_records=None,
    )
    save_journal_id_map(journal_id_map, "journal_id_map.json")
    print("Metadata Processing Complete!")
