"""
Embedding models:
- CBOW
- Skip-gram
- GloVe
- FastText
"""

try:
    from gensim.models import Word2Vec, FastText
    import gensim.downloader as api

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
  def train_word2vec(tokenized_corpus):
    """
    Train both CBOW and Skip-gram models.
    """

    cbow_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=0,
        epochs=100
    )

    skipgram_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,
        epochs=100
    )

    return cbow_model, skipgram_model
    def print_word2vec_results(model, word):
    """
    Print nearest neighbors.
    """

    print(f"\nNearest words for '{word}':")

    try:
        for w, score in model.wv.most_similar(word, topn=5):
            print(f"{w:<20} {score:.4f}")

    except KeyError:
        print("Word not found.")
      def train_fasttext(tokenized_corpus):
    """
    Train a FastText model.
    """

    model = FastText(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        epochs=100
    )

    return model
def load_glove():
    """
    Load pretrained GloVe model.
    """

    try:
        glove = api.load("glove-wiki-gigaword-50")
        return glove

    except Exception:
        return None
      def print_glove_results(glove, word):
    """
    Print nearest neighbors using GloVe.
    """

    if glove is None:
        print("GloVe model could not be loaded.")
        return

    try:
        for w, score in glove.most_similar(word, topn=5):
            print(f"{w:<20} {score:.4f}")

    except KeyError:
        print("Word not found.")
