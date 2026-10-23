# Information Retrieval and NLP

A simple educational Python project demonstrating fundamental **Information Retrieval (IR)** and **Natural Language Processing (NLP)** techniques.

This project was developed for the **Information Retrieval** course and provides practical implementations of classical retrieval models and modern embedding-based approaches.

---

## Features

- Boolean Retrieval Model
- TF-IDF Vector Space Model
- Word2Vec (CBOW)
- Word2Vec (Skip-gram)
- GloVe Embeddings
- FastText
- BERT Sentence Embeddings
- Transformer-based Embeddings (DistilBERT)

---

## Project Structure

```
Information-Retrieval-and-NLP/
├── data/
│   └── corpus.txt
├── notebooks/
│   └── IR_methods_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── boolean_retrieval.py
│   ├── tfidf_model.py
│   ├── embeddings.py
│   ├── bert_model.py
│   └── transformer_model.py
├── main.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Abdulbari-Alameer/Information-Retrieval-and-NLP.git

cd Information-Retrieval-and-NLP
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

There are two ways to run the project.

**1. Notebook: all seven methods side by side** (recommended)

Open [`notebooks/IR_methods_comparison.ipynb`](notebooks/IR_methods_comparison.ipynb) in Jupyter or Google Colab and run all cells. It ranks one corpus of 8 documents against the query *"neural networks for language understanding"* with Boolean retrieval, TF-IDF, Word2Vec (CBOW and Skip-gram), GloVe, FastText, Sentence-BERT and DistilBERT, and compares the results. The saved outputs can be read on GitHub without running anything.

**2. Command line: interactive search**

```bash
python main.py
```

The program loads `data/corpus.txt`, builds an inverted index and asks for a query. For each query it prints the Boolean retrieval matches and the TF-IDF ranking. Type `exit` to quit. The embedding models are in `src/` and are demonstrated in the notebook.

---

## Results

Top three documents for the query *"neural networks for language understanding"* (notebook outputs):

| Method | 1st | 2nd | 3rd | Score range (8 docs) |
|---|---|---|---|---|
| TF-IDF | D1 (0.43) | D2 (0.22) | D4 (0.11) | 0.00 – 0.43 |
| GloVe (mean of word vectors) | D1 (0.88) | D0 (0.87) | D2 (0.85) | 0.71 – 0.88 |
| Sentence-BERT | D6 (0.44) | D2 (0.43) | D5 (0.42) | 0.04 – 0.44 |
| DistilBERT (mean pooling) | D2 (0.86) | D1 (0.86) | D5 (0.83) | 0.69 – 0.86 |

D0 machine learning · D1 deep learning / neural networks · D2 natural language processing · D4 Python · D5 transformers / NLP · D6 word embeddings

- **TF-IDF** scores half of the documents at exactly 0 because they share no word with the query.
- **GloVe** and **DistilBERT** score every document above 0.68, so relevant and irrelevant documents are hard to tell apart.
- **Sentence-BERT**, trained for semantic similarity, separates them best: about 0.41–0.44 for the AI and language documents, 0.04–0.12 for the search and Python ones.
- **FastText** builds sensible vectors for unseen words (*learnt* → *learning*, *neurall* → *neural*).

The corpus is tiny on purpose, so these numbers illustrate how each method behaves; they are not a benchmark.

---

## Models Included

### Boolean Retrieval

- Exact keyword matching
- Supports:
  - AND
  - OR
  - NOT

---

### TF-IDF

- Term Frequency–Inverse Document Frequency
- Cosine Similarity ranking

---

### Word2Vec

- CBOW
- Skip-gram

---

### GloVe

- Pre-trained GloVe embeddings
- Demonstrates semantic similarity

---

### FastText

- Subword-based word embeddings
- Handles unseen words better than Word2Vec

---

### BERT

Uses the Sentence Transformers library to generate contextual sentence embeddings.

Model:

```
all-MiniLM-L6-v2
```

---

### Transformer

Uses Hugging Face Transformers with:

```
distilbert-base-uncased
```

Embeddings are generated using Mean Pooling.

---

## Technologies

- Python 3.10+
- NumPy
- Scikit-learn
- Gensim
- Sentence Transformers
- Hugging Face Transformers
- PyTorch

---

## Learning Objectives

This project demonstrates the differences between:

| Method | Description |
|----------|-------------|
| Boolean Retrieval | Exact keyword matching |
| TF-IDF | Weighted lexical matching |
| Word2Vec | Local semantic relationships |
| GloVe | Global word co-occurrence |
| FastText | Subword-based embeddings |
| BERT | Contextual sentence embeddings |
| Transformer | Deep contextual representations |

---

## License

This project is released under the MIT License.

---

## Author

**Abdulbari Al-Ameer**, Master's in Computer Science & Engineering, Tezpur University. Developed as an educational project for the Information Retrieval course.
