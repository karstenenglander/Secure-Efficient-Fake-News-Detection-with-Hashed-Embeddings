import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




# Load the trained model from the file
with open('Trained pkl/trained_RFHashHashVecHashEveryWord_embeddings1.pkl', 'rb') as file:
    X = pickle.load(file)

with open('Trained pkl/trained_RFHashHashVecHashEveryWord_labels1.pkl', 'rb') as file:
    y = pickle.load(file)

# Load the pickled model from the file
with open('trained_RF_HashVec_Hash_Trained_specific.pkl', 'rb') as file:
    model = pickle.load(file)

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
    'text': X.loc[y_test_subset.index, 'text'],
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

print("\n", "total time: ", elapsed_train_time)

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()