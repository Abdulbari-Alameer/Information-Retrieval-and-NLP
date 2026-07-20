"""
Boolean Retrieval Model
"""

from collections import defaultdict

from preprocessing import preprocess


def build_inverted_index(documents: list[str]) -> dict:
    """
    Build an inverted index from the document collection.
    """

    index = defaultdict(set)

    for doc_id, document in enumerate(documents):
        tokens = preprocess(document)

        for token in tokens:
            index[token].add(doc_id)

    return index


def get_postings(term: str, index: dict) -> set:
    """
    Return the posting list for a given term.
    """

    return index.get(term.lower(), set())
