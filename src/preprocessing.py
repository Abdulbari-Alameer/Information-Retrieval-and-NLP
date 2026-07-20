"""
Text preprocessing utilities.

This module provides simple preprocessing functions used
across all Information Retrieval models.
"""

import string

# Small stopword list for this educational project
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "is",
    "of",
    "the",
    "to",
}


def preprocess(text: str) -> list[str]:
    """
    Preprocess a single text document.

    Steps
    -----
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Tokenize using whitespace.
    4. Remove stopwords.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        List of processed tokens.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def preprocess_corpus(documents: list[str]) -> list[list[str]]:
    """
    Apply preprocessing to an entire document collection.

    Parameters
    ----------
    documents : list[str]
        Collection of text documents.

    Returns
    -------
    list[list[str]]
        Tokenized and cleaned documents.
    """

    return [
        preprocess(document)
        for document in documents
    ]


if __name__ == "__main__":

    sample_documents = [
        "Machine Learning is used in Artificial Intelligence.",
        "Natural Language Processing enables information retrieval.",
    ]

    print("Original Documents:\n")

    for document in sample_documents:
        print(document)

    print("\nProcessed Documents:\n")

    processed = preprocess_corpus(sample_documents)

    for tokens in processed:
        print(tokens)
