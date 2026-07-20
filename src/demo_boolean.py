"""
Demo for Boolean Retrieval.
"""

from utils import load_corpus
from boolean_retrieval import (
    build_inverted_index,
    boolean_search,
)

documents = load_corpus()

index = build_inverted_index(documents)

query = input("Enter Boolean query: ")

results = boolean_search(
    query=query,
    index=index,
    total_docs=len(documents),
)

print("\nMatching Documents")

if not results:
    print("No matching documents found.")

for doc_id in sorted(results):
    print(f"D{doc_id}: {documents[doc_id]}")
