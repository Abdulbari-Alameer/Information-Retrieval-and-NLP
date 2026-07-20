"""
Boolean Retrieval Model
"""

from collections import defaultdict

from src.preprocessing import preprocess


def build_inverted_index(documents: list[str]) -> dict[str, set[int]]:
    """
    Build an inverted index from a collection of documents.
    """

    index = defaultdict(set)

    for doc_id, document in enumerate(documents):
        tokens = preprocess(document)

        for token in tokens:
            index[token].add(doc_id)

    return index


def get_postings(term: str, index: dict[str, set[int]]) -> set[int]:
    """
    Return the posting list for a given term.
    """

    return index.get(term.lower(), set())


def boolean_search(
    query: str,
    index: dict[str, set[int]],
    total_docs: int,
) -> set[int]:
    """
    Execute a Boolean query.

    Supported formats:
        term
        term1 AND term2
        term1 OR term2
        NOT term
    """

    tokens = query.lower().split()

    if not tokens:
        return set()

    all_docs = set(range(total_docs))

    # Single term
    if len(tokens) == 1:
        return get_postings(tokens[0], index)

    # NOT term
    if len(tokens) == 2 and tokens[0] == "not":
        return all_docs - get_postings(tokens[1], index)

    # term1 AND term2
    if len(tokens) == 3 and tokens[1] == "and":
        left = get_postings(tokens[0], index)
        right = get_postings(tokens[2], index)
        return left & right

    # term1 OR term2
    if len(tokens) == 3 and tokens[1] == "or":
        left = get_postings(tokens[0], index)
        right = get_postings(tokens[2], index)
        return left | right

    raise ValueError(
        "Supported queries: "
        "'term', 'term1 AND term2', 'term1 OR term2', or 'NOT term'."
    )
