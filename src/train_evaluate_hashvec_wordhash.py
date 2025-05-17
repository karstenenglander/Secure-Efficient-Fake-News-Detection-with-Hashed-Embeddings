import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
import hashlib
from nltk.tokenize import TweetTokenizer
import spacy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken




# Load the separate CSV files
fake_data = pd.read_csv('data/ISOT_Fake.csv')
real_data = pd.read_csv('data/ISOT_True.csv')

# Use only the first n_elements of articles
#n_elements = 10000
#fake_data = fake_data.head(n_elements)
#real_data = real_data.head(n_elements)

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


# Tokenize the cleaned text
data['tokens'] = data['cleaned_text'].apply(word_tokenize)

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


def estimate_tokens(text):
    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Return the number of tokens
    return len(tokens)

# Apply this to all articles in your dataset and store the token counts
data['num_tokens'] = data['hashed_text'].apply(estimate_tokens)

# Calculate the average token count
average_token_count = data['num_tokens'].mean()
print(f"Average token count: {average_token_count}")

# Initialize the HashingVectorizer
vectorizer = HashingVectorizer(stop_words=None)

start_embedding_time = time.time()

# Prepare Data for Training
X = vectorizer.fit_transform(data['hashed_text'])

end_embedding_time = time.time()
elapsed_embedding_time = end_embedding_time - start_embedding_time
print(f"Elapsed embed time: {elapsed_embedding_time} seconds")

y = data['label']  # Ensure your dataset has the 'label' column

#with open('Trained pkl/trained_RFHashHashVecHashEveryWord_embeddings1.pkl', 'wb') as file:
    #pickle.dump(X, file)

#with open('Trained pkl/trained_RFHashHashVecHashEveryWord_labels1.pkl', 'wb') as file:
    #pickle.dump(y, file)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

start_train_time = time.time()
# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=2, min_samples_leaf=1, max_features=0.5, max_depth=30, criterion='entropy', bootstrap=False)
model.fit(X_train, y_train)
end_train_time = time.time()
elapsed_train_time = end_train_time - start_train_time
print(f"Elapsed train time: {elapsed_train_time} seconds")

#Select a subset of the test data
num_articles = 10000  # Specify the number of articles you want to use for testing
X_test_subset = X_test[:num_articles]
y_test_subset = y_test[:num_articles]

start_test_time = time.time()

# Predict and Evaluate
y_pred_proba = model.predict_proba(X_test_subset)
# Use the loaded model to make predictions
y_pred = model.predict(X_test_subset)

end_test_time = time.time()
elapsed_test_time = end_test_time - start_test_time
print(f"Elapsed test time: {elapsed_test_time} seconds")

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
with open('data/trained_RF_HashVecTrained.pkl', 'wb') as file:
    pickle.dump(model, file)

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()