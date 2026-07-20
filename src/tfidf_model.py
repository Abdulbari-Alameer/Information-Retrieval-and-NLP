"""
TF-IDF Vector Space Model
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(
    documents: list[str],
    query: str
) -> list[tuple[int, float]]:
    """
    Rank documents using TF-IDF and cosine similarity.
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    ranked_results = sorted(
        enumerate(similarity_scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results


def print_tfidf_results(
    documents: list[str],
    query: str
) -> None:
    """
    Display ranked TF-IDF search results.
    """

    print("\n===== TF-IDF RESULTS =====")
    print(f"Query: {query}\n")

    results = tfidf_search(documents, query)

    for doc_id, score in results:

        print(f"D{doc_id} | Score = {score:.4f}")
        print(documents[doc_id])
        print("-" * 60)
