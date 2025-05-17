import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import time
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import spacy
from nltk.tokenize import TweetTokenizer




# Load the separate CSV files
fake_data = pd.read_csv('data/ISOT_Fake.csv')
real_data = pd.read_csv('data/ISOT_True.csv')

# Use only the first n_elements of articles
n_elements = 10
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

def tokenize_text(text):
    return tweet_tokenizer.tokenize(text)


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


def hashWords(text):
    # Split the text into words
    words = tweet_tokenizer.tokenize(text)

    # List to hold hashed words
    hashed_words = []

    # Iterate over each word and hash it
    for word in words:
        hash_object = hashlib.sha256()
        hash_object.update(word.encode())
        hashed_word = hash_object.hexdigest()
        hashed_words.append(hashed_word)
    hashed_words_string = ' '.join(hashed_words)
    return hashed_words_string

data['hashed_text'] = data['cleaned_text'].apply(hashWords)

data['hashed_tokens'] = data['hashed_text'].apply(tokenize_text)

start_embedding_time = time.time()

# Create a list of tagged documents
tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data['hashed_tokens'])]

# Train a Doc2Vec model
doc2vec_model = Doc2Vec(tagged_documents, vector_size=50, window=2, min_count=1, workers=4)

# Generate embeddings
data['hashed_embedding'] = data['hashed_tokens'].apply(doc2vec_model.infer_vector)

end_embedding_time = time.time()
elapsed_embedding_time = end_embedding_time - start_embedding_time
print(f"Elapsed embed time: {elapsed_embedding_time} seconds")

with open('trained_RFHashD2VHashEveryWord_embeddings.pkl', 'wb') as file:
    pickle.dump(data, file)

# Prepare Data for Training
X = np.array(data['hashed_embedding'].tolist())
y = data['label']  # Ensure your dataset has the 'label' column

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

start_train_time = time.time()
# Train Model
model = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_split=5, min_samples_leaf=1, max_features='log2', max_depth=50, criterion='entropy', bootstrap=False)
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


# Save the trained model to a file
with open('trained_RF_D2V_Hash.pkl', 'wb') as file:
    pickle.dump(model, file)