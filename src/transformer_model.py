"""
Transformer Embedding Model using Hugging Face
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


MODEL_NAME = "distilbert-base-uncased"


def mean_pooling(model_output, attention_mask):
    """
    Compute sentence embeddings using mean pooling.
    """

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return (
        (token_embeddings * input_mask_expanded).sum(1)
        / input_mask_expanded.sum(1)
    )


def transformer_search(
    documents: list[str],
    query: str
) -> list[tuple[int, float]]:
    """
    Rank documents using DistilBERT embeddings.
    """

    if not TRANSFORMER_AVAILABLE:
        raise ImportError(
            "transformers is not installed."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    texts = documents + [query]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pooling(
        outputs,
        encoded["attention_mask"]
    )

    embeddings = embeddings.cpu().numpy()

    query_embedding = embeddings[-1]
    document_embeddings = embeddings[:-1]

    scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    ranked_results = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results


def print_transformer_results(
    documents: list[str],
    query: str
) -> None:
    """
    Print ranked Transformer search results.
    """

    print("\n===== TRANSFORMER RESULTS =====")
    print(f"Query: {query}\n")

    results = transformer_search(documents, query)

    for doc_id, score in results:
        print(f"D{doc_id} | Score = {score:.4f}")
        print(documents[doc_id])
        print("-" * 60)
