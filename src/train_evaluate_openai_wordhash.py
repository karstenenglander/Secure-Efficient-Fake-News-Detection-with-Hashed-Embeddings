import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
import threading
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import hashlib
from nltk.tokenize import TweetTokenizer


# Load the separate CSV files
fake_data = pd.read_csv('data/ISOT_Fake.csv')
real_data = pd.read_csv('data/ISOT_True.csv')

# Use only the first n_elements of articles

n_elements = 50
fake_data = fake_data.head(n_elements)
real_data = real_data.head(n_elements)

# Add 'label' column
fake_data['label'] = 'fake'
real_data['label'] = 'real'

# Concatenate the two DataFrames
data = pd.concat([fake_data, real_data], ignore_index=True)

# Define the nltk data directory
nltk_data_dir = os.path.expanduser('~/nltk_data')

# Check if 'stopwords' is downloaded
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora/stopwords')):
    nltk.download('stopwords')

# Check if 'punkt' is downloaded
if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt')):
    nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
tweet_tokenizer = TweetTokenizer()

# Preprocess the Data
def preprocess_text(text):
    # Process the text with SpaCy to identify named entities
    doc = nlp(text)

    # Preserve named entities by marking them (e.g., concatenating entity words with an underscore)
    preserved_entities = {ent.text: ent.text.replace(' ', '_') for ent in doc.ents}

    # Replace entities in the original text with their preserved forms
    for original, preserved in preserved_entities.items():
        text = text.replace(original, preserved)

    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Setup OpenAI API
openai.api_key = 'open-ai-key'

# Initialize embedding_count at the global scope
embedding_count = 0

# Generate Embeddings
embeddings = []


def optimized_split_into_chunks(text, chunk_size=8192):
    # Base case: If the text is shorter than the chunk size, return it as the only chunk
    if len(text) < chunk_size:
        return [text]

    chunks = []
    while text:
        # If the remaining text is shorter than or equal to the chunk size, add it as a chunk and break
        if len(text) < chunk_size:
            chunks.append(text)
            break

        # Find the last space within the chunk size to avoid splitting words
        split_point = text.rfind(' ', 0, chunk_size)

        # If no space is found (a very long word), use the chunk size as the split point
        if split_point == -1:
            split_point = chunk_size

        # Add the chunk up to the split point
        chunks.append(text[:split_point])

        # Remove the processed chunk from the text
        text = text[split_point:].lstrip()  # Remove leading spaces for the next chunk

    return chunks

def get_embedding_with_retry(text, model="text-embedding-ada-002", max_retries=10, backoff_factor=1):
    global embedding_count
    # Tokenize the text into words
    words = tweet_tokenizer.tokenize(text)

    # List to hold hashed words
    hashed_words = []

    # Iterate over each word and hash it
    for word in words:
        hash_object = hashlib.sha256()
        hash_object.update(word.encode())
        hashed_word = hash_object.hexdigest()
        hashed_words.append(hashed_word)

    # Join hashed words to form the final text to be chunked and embedded
    hashed_text = ' '.join(hashed_words)

    if len(hashed_text) < 8192:
        text_chunks = [hashed_text]
    else:
        print("-")
        text_chunks = list(optimized_split_into_chunks(hashed_text))

    chunk_embeddings = []
    for chunk in text_chunks:
        for i in range(max_retries):
            try:
                response = openai.Embedding.create(input=chunk, model=model)
                chunk_embeddings.append(response['data'][0]['embedding'])
                embedding_count += 1
                break  # Exit retry loop on success
            except (openai.error.ServiceUnavailableError, openai.error.APIConnectionError, openai.error.APIError) as e:
                if i < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** i)
                    time.sleep(sleep_time)
                else:
                    raise  # Re-raise the last exception if it's still failing
    if chunk_embeddings:
        return np.mean(chunk_embeddings, axis=0)
    else:
        return None

total_articles = len(data)
processed_articles = 0

def log_progress():
    while processed_articles < total_articles:
        time.sleep(10)
        progress_percentage = (processed_articles / total_articles) * 100
        print(f"{progress_percentage:.2f}% of articles processed")

# Start progress logging in a separate thread
progress_thread = threading.Thread(target=log_progress)
progress_thread.daemon = True  # Ensure the thread exits when the main program exits
progress_thread.start()

start_embedding_time = time.time()

# Generate embeddings with progress tracking
for idx, row in data.iterrows():
    embeddings.append(get_embedding_with_retry(row['cleaned_text']))
    processed_articles += 1

data['embedding'] = embeddings

#with open('trained_RFHashOpenAIHashFinalTweetNER_embeddings.pkl', 'wb') as file:
    #pickle.dump(data, file)

end_embedding_time = time.time()
elapsed_embedding_time = end_embedding_time - start_embedding_time
print(f"Elapsed embed time: {elapsed_embedding_time} seconds")

# Stop the progress logging
progress_thread.join()

# Prepare Data for Training
X = np.array(data['embedding'].tolist())
y = data['label']  # Ensure your dataset has the 'label' column

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

start_train_time = time.time()
# Train Model
model = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', max_depth=50, criterion='entropy', bootstrap=False)
model.fit(X_train, y_train)
end_train_time = time.time()
elapsed_train_time = end_train_time - start_train_time
print(f"Elapsed train time: {elapsed_train_time} seconds")

# Predict and Evaluate
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Extract probabilities for the 'fake' class
fake_probabilities = y_pred_proba[:, model.classes_.tolist().index('fake')]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='fake')
recall = recall_score(y_test, y_pred, pos_label='fake')
f1 = f1_score(y_test, y_pred, pos_label='fake')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Output Scores for Each Article
results = pd.DataFrame({
    'text': data.loc[y_test.index, 'text'],
    'predicted_label': y_pred,
    'fake_probability': fake_probabilities
})

print(results)
print("Greater than 0.5 probability is fake, less than or equal to 0.5 probability is real")

# Convert actual labels to binary
y_test_binary = (y_test == 'fake').astype(int)

# Calculate absolute differences
differences = np.abs(fake_probabilities - y_test_binary)

# Calculate average difference
average_difference = np.mean(differences)

print(f"Average difference between predicted probabilities and actual labels: {average_difference}")

print("\n", "total time: ", elapsed_train_time + elapsed_embedding_time)
print("\n", "total embeddings: ", embedding_count)