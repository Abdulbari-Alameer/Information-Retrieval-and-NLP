"""
Utility functions for the Information Retrieval project.
"""

from pathlib import Path


# Path to the corpus file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.txt"


def load_corpus() -> list[str]:
    """
    Load the document collection from the corpus file.

    Returns
    -------
    list[str]
        A list of non-empty documents.
    """

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Corpus file not found:\n{CORPUS_FILE}"
        )

    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        documents = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return documents


def print_corpus(documents: list[str]) -> None:
    """
    Print the loaded corpus with document IDs.
    """

    print("\nCorpus\n" + "-" * 50)

    for doc_id, document in enumerate(documents):
        print(f"D{doc_id}: {document}")


if __name__ == "__main__":

    corpus = load_corpus()

    print(f"Loaded {len(corpus)} documents.")

    print_corpus(corpus)
