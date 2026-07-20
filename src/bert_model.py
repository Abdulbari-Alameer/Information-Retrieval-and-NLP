"""
BERT Sentence Embedding Model
"""

from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


MODEL_NAME = "all-MiniLM-L6-v2"


def bert_search(
    documents: list[str],
    query: str
) -> list[tuple[int, float]]:
    """
    Rank documents using Sentence-BERT embeddings.
    """

    if not SBERT_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed."
        )

    model = SentenceTransformer(MODEL_NAME)

    document_embeddings = model.encode(
        documents,
        convert_to_numpy=True
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_results = sorted(
        enumerate(similarity_scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results


def print_bert_results(
    documents: list[str],
    query: str
) -> None:
    """
    Print ranked BERT search results.
    """

    print("\n===== BERT RESULTS =====")
    print(f"Query: {query}\n")

    results = bert_search(documents, query)

    for doc_id, score in results:
        print(f"D{doc_id} | Score = {score:.4f}")
        print(documents[doc_id])
        print("-" * 60)
