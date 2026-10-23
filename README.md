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

Run the main program:

```bash
python main.py
```

The program will:

- Load the corpus
- Build the Boolean Retrieval index
- Execute Boolean Retrieval
- Rank documents using TF-IDF
- Train CBOW and Skip-gram models
- Train FastText
- Demonstrate GloVe embeddings
- Compute BERT sentence similarities
- Compute Transformer-based document similarities

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
