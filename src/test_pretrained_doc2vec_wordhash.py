import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle




# Load the trained model from the file
with open('Trained pkl/trained_RFHashD2VHashEveryWord_embeddings.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the pickled model from the file
with open('trained_RF_D2V_Hash_trained_specific.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare Data for Training
X = np.array(loaded_model['hashed_embedding'].tolist())
y = loaded_model['label']  # Ensure your dataset has the 'label' column

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Select a subset of the test data
num_articles = 40000 # Specify the number of articles you want to use for testing
X_test_subset = X_test[:num_articles]
y_test_subset = y_test[:num_articles]

start_train_time = time.time()

# Predict and Evaluate
y_pred_proba = model.predict_proba(X_test_subset)
y_pred = model.predict(X_test_subset)

end_train_time = time.time()
elapsed_train_time = end_train_time - start_train_time
print(f"Elapsed test time: {elapsed_train_time} seconds")

# Extract probabilities for the 'fake' class
fake_probabilities = y_pred_proba[:, model.classes_.tolist().index('fake')]

# Evaluate
accuracy = accuracy_score(y_test_subset, y_pred)
precision = precision_score(y_test_subset, y_pred, pos_label='fake')
recall = recall_score(y_test_subset, y_pred, pos_label='fake')
f1 = f1_score(y_test_subset, y_pred, pos_label='fake')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Output Scores for Each Article
results = pd.DataFrame({
    'text': loaded_model.loc[y_test.index, 'text'],
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



