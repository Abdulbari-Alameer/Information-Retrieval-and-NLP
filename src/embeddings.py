"""
Embedding Models
----------------
CBOW
Skip-gram
FastText
GloVe
"""

try:
    from gensim.models import Word2Vec, FastText
    import gensim.downloader as api

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


def train_word2vec(tokenized_corpus: list[list[str]]):
    """
    Train CBOW and Skip-gram models.
    """

    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is not installed.")

    cbow_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=0,
        epochs=100,
        workers=1,
        seed=42,
    )

    skipgram_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,
        epochs=100,
        workers=1,
        seed=42,
    )

    return cbow_model, skipgram_model


def train_fasttext(tokenized_corpus: list[list[str]]):
    """
    Train a FastText model.
    """

    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is not installed.")

    model = FastText(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        epochs=100,
        workers=1,
        seed=42,
    )

    return model


def load_glove():
    """
    Load pretrained GloVe embeddings.
    """

    if not GENSIM_AVAILABLE:
        return None

    try:
        return api.load("glove-wiki-gigaword-50")
    except Exception:
        return None


def print_word_neighbors(model, word: str, topn: int = 5):
    """
    Print nearest words.
    """

    print(f"\nNearest neighbors for '{word}':")

    try:
        neighbors = model.wv.most_similar(word, topn=topn)

        for neighbor, score in neighbors:
            print(f"{neighbor:<20} {score:.4f}")

    except KeyError:
        print("Word not found in vocabulary.")


def print_glove_neighbors(glove_model, word: str, topn: int = 5):
    """
    Print nearest words using GloVe.
    """

    if glove_model is None:
        print("GloVe model is unavailable.")
        return

    print(f"\nNearest neighbors for '{word}':")

    try:
        neighbors = glove_model.most_similar(word, topn=topn)

        for neighbor, score in neighbors:
            print(f"{neighbor:<20} {score:.4f}")

    except KeyError:
        print("Word not found in GloVe vocabulary.")
