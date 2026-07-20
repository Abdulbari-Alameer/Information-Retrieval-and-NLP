"""
BERT Sentence Embeddings using Sentence Transformers
"""

from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


def bert_search(documents, query):
    """
    Rank documents using Sentence-BERT embeddings.
    """

    if not SBERT_AVAILABLE:
        print("sentence-transformers is not installed.")
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")

    document_embeddings = model.encode(documents)

    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_results = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results


def print_bert_results(documents, query):
    """
    Print ranked BERT search results.
    """

    results = bert_search(documents, query)

    if not results:
        return

    print("\n===== BERT RESULTS =====")
    print(f"Query: {query}\n")

    for doc_id, score in results:
        print(f"D{doc_id} | Score = {score:.4f}")
        print(documents[doc_id])
        print("-" * 60)
