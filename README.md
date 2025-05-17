# Secure & Efficient Fake News Detection with Hashed Embeddings 🛡️📰

## Overview

This repository contains the code and findings from a research project focused on developing a robust and privacy-conscious fake news detection model. The primary model, utilizing a RandomForest classifier with OpenAI text embeddings and SHA-256 hashed word features, achieved an accuracy of **99.89%** on benchmark datasets (e.g., Lifferth/ISOT - specify which one gave this result). This project was part of a Cybersecurity REU at Montclair State University and was presented at the MIT URTC IEEE Conference (2024).

The framework explores various text preprocessing pipelines, vectorization techniques (OpenAI embeddings, Doc2Vec, HashingVectorizer), and model optimization strategies, with a key emphasis on computational efficiency and data anonymization for potential deployment in Online Social Networks (OSNs).

## The Challenge: Combating Misinformation Securely

The proliferation of fake news on social media poses a significant threat. While many detection models exist, deploying them directly within OSNs raises privacy concerns due to the sensitive nature of user-generated content. This project aimed to:
1.  Achieve state-of-the-art accuracy in fake news detection.
2.  Incorporate a layer of data anonymization through cryptographic hashing without significantly compromising model performance.
3.  Explore and compare different text embedding and vectorization techniques for this task.
4.  Optimize model parameters for peak performance.

## Key Features & Methodologies

*   **High-Accuracy Model:** Achieved 99.89% accuracy using a RandomForest classifier, outperforming many existing approaches.
*   **Advanced Text Preprocessing:**
    *   NLTK and SpaCy for tokenization, stop-word removal, and lemmatization.
    *   Named Entity Recognition (NER) preservation to retain important contextual information.
    *   Specialized `TweetTokenizer` for handling social media text nuances.
*   **Privacy-Enhancing Hashing:**
    *   Implemented SHA-256 hashing of individual words *before* embedding/vectorization. This provides a level of anonymization, making it harder to reverse-engineer original text from the features while still allowing the model to learn patterns.
*   **Diverse Vectorization Techniques Explored:**
    *   **OpenAI Embeddings (`text-embedding-ada-002`):** Leveraged powerful pre-trained language model embeddings, often yielding the best performance. Handled API rate limits and chunking for long texts.
    *   **Doc2Vec:** Trained custom document embeddings to capture semantic meaning.
    *   **HashingVectorizer:** A memory-efficient technique for converting text to numerical features.
*   **Efficient Model & Training:**
    *   Utilized RandomForest, known for its robustness and efficiency.
    *   Optimized hyperparameters using `RandomizedSearchCV`.
    *   Pickled trained models and pre-computed embeddings for faster iteration and deployment.
*   **Comprehensive Evaluation:**
    *   Metrics: Accuracy, Precision, Recall, F1-Score.
    *   Confusion matrices for detailed performance analysis.
    *   Analysis of prediction probabilities and average difference from actual labels.
*   **Cross-Dataset Evaluation:** Tested models on multiple datasets (ISOT, Lifferth) to assess generalizability.

## Project Structure

*   `/data/`: Contains datasets used for training and testing (or instructions to acquire them).
*   `/src/`: Source code for the project.
    *   `train_evaluate_best_model.py`: Script to train and evaluate the top-performing model (OpenAI embeddings with hashed words).
    *   `predict_new_article.py`: Example script to classify a new piece of text.
    *   `/experiments/`: Contains scripts for other vectorization methods and dataset combinations (Doc2Vec, HashingVectorizer).
    *   `/tuning/`: Scripts used for hyperparameter optimization.
*   `/saved_models_embeddings/`: Stores pre-trained model (.pkl) and pre-computed embedding files.
*   `/results/`: (Optional) Output files like confusion matrices, performance tables.
*   `README.md`: This file.

## How to Run (Example for the best model)

1.  **Setup:**
    *   Clone the repository.
    *   Install dependencies: `pip install pandas nltk gensim scikit-learn spacy openai seaborn matplotlib tiktoken`
    *   Download NLTK resources: `python -m nltk.downloader stopwords punkt`
    *   Download SpaCy model: `python -m spacy download en_core_web_sm`
    *   Place datasets in the `/data/` directory.
    *   (If using OpenAI embeddings) Set your OpenAI API key, e.g., as an environment variable or directly in the script (for local testing only, not recommended for public repos).
2.  **Training & Evaluation:**
    *   Navigate to the `src/` directory.
    *   Run the main training script: `python train_evaluate_best_model.py`
    *   This will preprocess data, generate/load embeddings, train the RandomForest model, and print evaluation metrics.
3.  **Prediction on New Text:**
    *   Run the prediction script: `python predict_new_article.py` (You might need to modify it to take text input).

*(Adjust the "How to Run" section based on how you structure your final main script and if embeddings/models need to be generated first or are loaded from `/saved_models_embeddings/`)*

## Key Findings & Learnings

*   **Hashing Impact:** Applying SHA-256 hashing to individual words before embedding provided a good balance between data anonymization and model performance. While there might be a slight performance trade-off compared to non-hashed embeddings, the high accuracy achieved indicates its viability.
*   **OpenAI Embeddings Superiority:** Models using OpenAI's `text-embedding-ada-002` generally outperformed Doc2Vec and HashingVectorizer for this task, likely due to the richness of the pre-trained representations.
*   **Preprocessing Importance:** A careful preprocessing pipeline, including NER preservation and appropriate tokenization, is crucial for effective feature extraction.
*   **Hyperparameter Tuning:** Significantly impacts RandomForest performance. `RandomizedSearchCV` was effective in finding optimal settings.
*   **Computational Considerations:** HashingVectorizer is the most memory-efficient, while OpenAI embeddings require API calls and can be slower to generate initially. Doc2Vec training time depends on the dataset size.

This project demonstrates a practical and effective approach to building high-accuracy, privacy-aware fake news detection systems, a critical need in today's information ecosystem.
