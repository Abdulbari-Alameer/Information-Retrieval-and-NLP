"""
Text preprocessing utilities.

This module provides simple preprocessing functions used
across all Information Retrieval models.
"""

import string

# A small set of stopwords for this toy project
STOPWORDS = {
    "is",
    "in",
    "the",
    "and",
    "to",
    "for",
    "of",
    "a",
    "an"
}


def preprocess(text: str) -> list[str]:
    """
    Convert text into cleaned tokens.

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords

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

    # Split into words
    tokens = text.split()

    # Remove stopwords
    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return tokens
