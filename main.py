"""
Main entry point for the Information Retrieval project.
"""

from src.utils import load_corpus
from src.boolean_retrieval import (
    build_inverted_index,
    boolean_search,
)


def main():
    """Run the Information Retrieval demo."""

    documents = load_corpus()

    print("=" * 60)
    print("Information Retrieval and NLP Toolkit")
    print("=" * 60)

    print(f"\nLoaded {len(documents)} documents.")

    index = build_inverted_index(documents)

    while True:
        print("\nBoolean Retrieval")
        query = input("Enter a Boolean query (or 'exit'): ").strip()

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        try:
            results = boolean_search(
                query=query,
                index=index,
                total_docs=len(documents),
            )

            if not results:
                print("\nNo matching documents found.")
                continue

            print("\nMatching Documents:\n")

            for doc_id in sorted(results):
                print(f"D{doc_id}: {documents[doc_id]}")

        except ValueError as error:
            print(error)


if __name__ == "__main__":
    main()
