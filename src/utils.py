"""
Utility functions for the Information Retrieval project.
"""

from pathlib import Path


def load_corpus(file_path: str = "data/corpus.txt") -> list[str]:
    """
    Load the document collection from a text file.

    Each non-empty line is treated as a separate document.

    Parameters
    ----------
    file_path : str
        Path to the corpus file.

    Returns
    -------
    list[str]
        List of documents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {file_path}"
        )

    with path.open("r", encoding="utf-8") as file:
        documents = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return documents
