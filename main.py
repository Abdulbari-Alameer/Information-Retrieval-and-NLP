"""
Main entry point for the Information Retrieval and NLP project.
"""

from src.utils import load_corpus, print_corpus
from src.boolean_retrieval import (
    build_inverted_index,
    boolean_search,
)
from src.tfidf_model import print_tfidf_results


def main():

    print("=" * 60)
    print("Information Retrieval and NLP")
    print("=" * 60)

    # Load documents
    documents = load_corpus()

    print(f"\nLoaded {len(documents)} documents.")

    print_corpus(documents)

    # Build Boolean index
    index = build_inverted_index(documents)

    while True:

        print("\n" + "=" * 60)

        query = input(
            "Enter a query (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        # ---------------- Boolean Retrieval ----------------

        print("\nBOOLEAN RETRIEVAL")
        print("-" * 60)

        try:

            results = boolean_search(
                query=query,
                index=index,
                total_docs=len(documents),
            )

            if results:

                for doc_id in sorted(results):
                    print(f"D{doc_id}: {documents[doc_id]}")

            else:
                print("No matching documents.")

        except ValueError as error:
            print(error)

        # ---------------- TF-IDF ----------------

        print("\nTF-IDF")
        print("-" * 60)

        print_tfidf_results(
            documents,
            query
        )


if __name__ == "__main__":
    main()
