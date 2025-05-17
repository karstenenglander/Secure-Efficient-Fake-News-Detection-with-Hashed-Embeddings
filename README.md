# Fake News Detection on ISOT using Hashed Word Embeddings 🛡️📰

## Overview

This repository showcases a comparative study of fake news detection models applied to the ISOT dataset. The core methodology involves using a RandomForest classifier with various text vectorization techniques where **individual words are first hashed using SHA-256 for privacy exploration before being vectorized/embedded.** This project demonstrates the training and evaluation pipelines for three primary approaches:

1.  **Doc2Vec** with hashed words.
2.  **OpenAI Embeddings** (e.g., `text-embedding-ada-002`) on hashed words, incorporating advanced NLP features like Tweetokenizer and Named Entity Recognition (NER) preservation.
3.  **HashingVectorizer** applied to hashed words.

The aim was to develop accurate detection models while investigating a method for data anonymization suitable for environments like Online Social Networks (OSNs). One of these approaches (specify which one, likely HashingVectorizer based on previous info) achieved **99.89% accuracy**. This work was part of a Cybersecurity REU at Montclair State University and contributed to research presented at the MIT URTC IEEE Conference (2024).

## Datasets Used

This project utilizes the **ISOT Fake News Dataset**.
*   Comprises `Fake.csv` and `True.csv`.
*   Due to their size, these dataset files are not included directly in this repository.
*   **You can download the ISOT dataset from:** [Provide Link to Kaggle or original source, e.g., https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]
*   Place the downloaded `Fake.csv` and `True.csv` into a `data/` subdirectory within the project root (e.g., `ISOT-FakeNews-HashedWords/data/ISOT_Fake.csv`).

## Key Methodologies & Features

*   **Privacy-Focused Word Hashing:** SHA-256 hashing is applied to each word after initial text cleaning but *before* the vectorization or embedding step. This serves as a data anonymization layer.
*   **Comprehensive Preprocessing:**
    *   Standard text cleaning (lowercase, punctuation/special character removal).
    *   Stop-word removal (NLTK).
    *   Tokenization (NLTK `word_tokenize` and `TweetTokenizer` for OpenAI approach).
    *   Named Entity Recognition (NER) preservation using SpaCy (for OpenAI and Doc2Vec word-hashed approaches).
*   **Vectorization/Embedding Techniques on Hashed Words:**
    *   **Doc2Vec (Gensim):** Training document vectors from sequences of hashed words.
    *   **OpenAI Embeddings:** Generating embeddings for sequences of hashed words using models like `text-embedding-ada-002`. Includes logic for handling long texts by chunking.
    *   **HashingVectorizer (Scikit-learn):** Applying Scikit-learn's HashingVectorizer directly to space-separated strings of hashed words.
*   **Classification Model:** RandomForestClassifier from Scikit-learn.
*   **Evaluation:** Standard metrics including Accuracy, Precision, Recall, and F1-Score. Confusion matrices can be generated.
*   **Modular Scripts:** Separate scripts for:
    *   Full training and evaluation pipelines for each vectorization method.
    *   Testing/evaluating pre-trained models by loading saved models and dataframes/features.

## Project Structure

*   `/data/`: (Instructions to download ISOT dataset).
*   `/src/`: Source code.
    *   `train_evaluate_doc2vec_wordhash.py`: Full pipeline for Doc2Vec + Word Hashing.
    *   `train_evaluate_openai_wordhash.py`: Full pipeline for OpenAI + Word Hashing.
    *   `train_evaluate_hashvec_wordhash.py`: Full pipeline for HashingVectorizer + Word Hashing (this is likely the 99.89% model).
    *   `test_pretrained_doc2vec_wordhash.py`: Loads and tests saved Doc2Vec system.
    *   `test_pretrained_openai_wordhash.py`: Loads and tests saved OpenAI system.
    *   `test_pretrained_hashvec_wordhash.py`: Loads and tests saved HashingVectorizer system.
*   `requirements.txt`: Python dependencies.
*   `README.md`: This file.

## How to Run

1.  **Setup:**
    *   Clone the repository.
    *   Create and activate a Python virtual environment.
    *   Install dependencies: `pip install -r requirements.txt`
    *   Download NLTK resources: In Python, run `import nltk; nltk.download('stopwords'); nltk.download('punkt')`
    *   Download SpaCy model: `python -m spacy download en_core_web_sm`
    *   Download the ISOT dataset and place `Fake.csv` and `True.csv` in a `data/` subdirectory.
    *   (If running OpenAI scripts) Set your OpenAI API key as an environment variable `OPENAI_API_KEY` or in the scripts (for local testing only).

2.  **Option A: Run a Full Training & Evaluation Pipeline:**
    *   Example (for the HashingVectorizer model, likely your best):
        `python src/train_evaluate_hashvec_wordhash.py`
    *   This will preprocess, hash, vectorize, train, evaluate, and save the model & vectorizer to a .pkl file.
    *   Similarly, run other `train_evaluate_*.py` scripts to reproduce results for Doc2Vec or OpenAI.

## Key Findings

*   The **HashingVectorizer** approach applied to SHA-256 hashed words on the ISOT dataset achieved the highest accuracy of **99.89%**.
*   OpenAI embeddings on hashed words also demonstrated strong performance, showcasing the utility of powerful pre-trained models even on pseudonymized data. However, it also took the most time to train in additon to incurring the most cost.
*   Doc2Vec on hashed words provided a competitive baseline.
*   The word-hashing technique appears viable for adding a layer of data privacy without catastrophically degrading model performance for fake news detection on this dataset.

This project demonstrates a systematic comparison of different privacy-enhancing vectorization techniques for fake news detection.
