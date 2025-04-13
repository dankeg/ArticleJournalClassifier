import json
import os
import time
from typing import Dict, List

import pymupdf  # or "import fitz as pymupdf"
import requests


def load_journal_id_map(
    input_file: str = "journal_id_map.json",
) -> Dict[str, List[str]]:
    """Loads a JSON file mapping journals to arXiv IDs.

    Args:
        input_file (str): Path to a JSON file structured as {journal: [arxiv_ids]}.

    Returns:
        Dict[str, List[str]]: A dictionary mapping journal names to lists of arXiv IDs.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def download_pdf(arxiv_id: str, save_path: str, download_delay: float = 1.0) -> bool:
    """Downloads a PDF from arXiv for a given arXiv ID.

    Args:
        arxiv_id (str): The arXiv identifier (e.g., "2301.12345").
        save_path (str): The file path where the PDF will be saved.
        download_delay (float): Download timeout, to prevent rate-limiting.

    Returns:
        bool: True if the PDF was successfully downloaded, False otherwise.
    """
    url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
    success = False
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            success = True
        else:
            print(
                f"Failed to download {arxiv_id} (status code: {response.status_code})."
            )
    except requests.RequestException as e:
        print(f"Exception downloading {arxiv_id}: {e}")
    finally:
        time.sleep(download_delay)
    return success


def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): Local path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    doc = pymupdf.open(pdf_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(texts)


def process_single_journal(
    journal: str, arxiv_ids: List[str], max_articles: int, download_delay: float = 1.0
) -> List[str]:
    """Processes a single journal by downloading PDFs and extracting text.

    For each arXiv ID, the PDF is downloaded and its text extracted.
    Processing stops after max_articles are processed.

    Args:
        journal (str): The journal name.
        arxiv_ids (List[str]): List of arXiv IDs for the journal.
        max_articles (int): Maximum number of articles to process for this journal.
        download_delay (float): Download timeout, to prevent rate-limiting.

    Returns:
        List[str]: A list of extracted texts for the journal.
    """
    print(
        f"Processing journal '{journal}' with {len(arxiv_ids)} articles (cap {max_articles})."
    )
    texts = []
    count = 0
    for arxiv_id in arxiv_ids:
        if count >= max_articles:
            break
        pdf_path = f"{arxiv_id}.pdf"
        if not download_pdf(arxiv_id, pdf_path, download_delay):
            continue
        try:
            text_content = extract_pdf_text(pdf_path)
            texts.append(text_content)
        except Exception as e:
            print(f"Failed to extract text from {pdf_path}: {e}")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        count += 1
        if count % 50 == 0:
            print(f"Journal '{journal}': processed {count} articles.")
    print(f"Finished journal '{journal}' with {count} articles.")
    return texts


def build_final_corpus(
    journal_id_map: Dict[str, List[str]],
    top_n: int,
    max_articles: int,
    download_delay: float,
) -> Dict[str, List[str]]:
    """Builds the final corpus by processing the top journals sequentially.

    Top journals are selected based on the number of articles.
    For each journal, up to max_articles PDFs are downloaded and their texts extracted.

    Args:
        journal_id_map (Dict[str, List[str]]): Mapping of journals to lists of arXiv IDs.
        top_n (int): Number of top journals to process.
        max_articles (int): Maximum number of articles per journal.
        download_delay (float): Download timeout, to prevent rate-limiting.

    Returns:
        Dict[str, List[str]]: A dictionary mapping journal names to lists of extracted text.
    """
    sorted_journals = sorted(
        journal_id_map.items(), key=lambda x: len(x[1]), reverse=True
    )[:top_n]
    print(f"Selected top {top_n} journals. Each capped at {max_articles} articles.")
    final_corpus = {}
    for journal, arxiv_ids in sorted_journals:
        texts = process_single_journal(journal, arxiv_ids, max_articles, download_delay)
        final_corpus[journal] = texts
    return final_corpus


def save_final_corpus(final_output_file: str, corpus: Dict[str, List[str]]) -> None:
    """Saves the final corpus dictionary to a JSON file.

    Args:
        final_output_file (str): The file path for the output JSON.
        corpus (Dict[str, List[str]]): Dictionary mapping journals to lists of extracted text.
    """
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved final corpus to {final_output_file}.")


def main() -> None:
    """Builds a PDF corpus from arXiv articles and saves it to a JSON file."""

    journal_id_map = load_journal_id_map("journal_id_map.json")
    top_n = 20
    max_articles = 1000

    # ArXiv rate limits you if you make requests that are too frequent.
    # Including this limits the chance of this, but doesn't eliminate it
    download_delay = 0.1
    final_output_file = "top_journals_text.json"

    final_corpus = build_final_corpus(
        journal_id_map, top_n, max_articles, download_delay
    )
    save_final_corpus(final_output_file, final_corpus)


if __name__ == "__main__":
    main()
