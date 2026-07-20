"""
Transformer Embeddings using Hugging Face Transformers
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
    """Compute sentence embedding using mean pooling."""

    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return (token_embeddings * mask).sum(1) / mask.sum(1)


def transformer_search(documents, query):
    """
    Rank documents using DistilBERT embeddings.
    """

    if not TRANSFORMER_AVAILABLE:
        print("transformers is not installed.")
        return []

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
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])

    embeddings = embeddings.cpu().numpy()

    query_embedding = embeddings[-1]
    document_embeddings = embeddings[:-1]

    scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


def print_transformer_results(documents, query):
    """
    Print ranked Transformer results.
    """

    results = transformer_search(documents, query)

    if not results:
        return

    print("\n===== TRANSFORMER RESULTS =====")
    print(f"Query: {query}\n")

    for doc_id, score in results:
        print(f"D{doc_id} | Score = {score:.4f}")
        print(documents[doc_id])
        print("-" * 60)
