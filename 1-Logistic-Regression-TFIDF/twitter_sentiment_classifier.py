# Import basic libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, confusion_matrix
import seaborn as sns

# Load datasets
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Handle missing values
train_df.dropna(subset=["Text", "Label"], inplace=True)
val_df.dropna(subset=["Text", "Label"], inplace=True)
test_df.dropna(subset=["Text"], inplace=True)

# EDA
# Class Distribution
print("Class Distribution (Raw Data):")
print(train_df['Label'].value_counts(normalize=True))
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Label', hue='Label', palette='viridis')
plt.title("Class Distribution (0: Negative, 1: Positive)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Raw Text Length Analysis (Characters)
train_df['raw_text_length'] = train_df['Text'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x='raw_text_length', hue='Label', bins=50, kde=True, palette='viridis')
plt.title("Raw Text Length Distribution (Characters)")
plt.xlabel("Number of Characters")
plt.show()

# Top Raw N-grams (Before Preprocessing)
def plot_raw_ngrams(text_series, title, ngram_range=(1, 1), top_n=20):
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=2000)
    X = vec.fit_transform(text_series)
    features = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1
    df = pd.DataFrame({'ngram': features, 'score': scores}).sort_values('score', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score', y='ngram', hue='ngram', data=df, palette='viridis')
    plt.title(f"Top {ngram_range[0]}-grams in {title}")
    plt.xlabel("TF-IDF Score")
    plt.show()

# Plot for Positive Class (Raw)
positive_raw = train_df[train_df['Label'] == 1]['Text']
plot_raw_ngrams(positive_raw, "Raw Positive Tweets", (1, 2))

# Plot for Negative Class (Raw)
negative_raw = train_df[train_df['Label'] == 0]['Text']
plot_raw_ngrams(negative_raw, "Raw Negative Tweets", (1, 2))

# Text Preprocessing 
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Achieve anonymization
    text = re.sub(r"@\S+", "X", text)
    text = re.sub(r"http\S+", "http", text)

    # Remove punctuation
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans(punctuation_to_remove, " " * len(punctuation_to_remove)))
    text = re.sub(r"\b(\w+)'(\w+)\b", r"\1\2", text)

    # Remove non-ASCII characters (fix mojibake issues)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove repeated characters (3 or more times → 1 occurrence)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Correct the spelling mistakes
    text = re.sub(r"\b(hapy)\b", "happy", text)
    text = re.sub(r"\b(angy)\b", "angry", text)
    text = re.sub(r"\b(luv)\b", "love", text)
    text = re.sub(r"\b(amzing)\b", "amazing", text)
    text = re.sub(r"\b(terible)\b", "terrible", text)
    text = re.sub(r"\b(excelent)\b", "excellent", text)
    text = re.sub(r"\b(performnce)\b", "performance", text)
    text = re.sub(r"\b(gud)\b", "good", text)
    text = re.sub(r"\b(vry)\b", "very", text)
    text = re.sub(r"\b(fantstic)\b", "fantastic", text)
    text = re.sub(r"\b(gr8)\b", "great", text)
    text = re.sub(r"\b(horrble)\b", "horrible", text)
    text = re.sub(r"\b(im)\b", "i am", text)
    text = re.sub(r"\b(omg)\b", "oh my god", text)
    text = re.sub(r"\b(plz)\b", "please", text)
    text = re.sub(r"\b(thx)\b", "thanks", text)

    return text

# Apply preprocessing
train_df["Text"] = train_df["Text"].apply(preprocess_text)
val_df["Text"] = val_df["Text"].apply(preprocess_text)
test_df["Text"] = test_df["Text"].apply(preprocess_text)

# Split the data
X_train, y_train = train_df["Text"], train_df["Label"]
X_val, y_val = val_df["Text"], val_df["Label"]
X_test = test_df["Text"]

# Feature Extraction - TF-IDF Vectorization
my_stop_words = [
    "the", "and", "is", "in", "it", "to", "of", "that", "this", 
    "a", "for", "on", "with", "as", "at", "by", "an", "be", "or", "from",
    "just"
]
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='my_stop_words', max_features=25000, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Use the best hyperparameters
best_hyperparameters = {'C': 2, 'solver': 'lbfgs', 'penalty': 'l2'}

# Train the final model with the best hyperparameters
best_model = LogisticRegression(**best_hyperparameters, max_iter=500, random_state=10)
best_model.fit(X_train_tfidf, y_train)

# Evaluate on the train set
y_train_pred = best_model.predict(X_train_tfidf)
print("\nTraining Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.6f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.6f}")
print(f"Recall: {recall_score(y_train, y_train_pred):.6f}")
print(f"F1-Score: {f1_score(y_train, y_train_pred):.6f}")

# Evaluate on the validation set
y_val_pred = best_model.predict(X_val_tfidf)
print("\nValidation Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.6f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.6f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.6f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.6f}")
print("\nClassification Report: \n", classification_report(y_val, y_val_pred, zero_division=0))

# Calculate the performance gap
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nPerformance Gap (Train - Val): {train_acc - val_acc:.4f}")

# Predictions on test set
y_test_pred = best_model.predict(X_test_tfidf)

# Create submission file
submission_df = pd.DataFrame({"ID": test_df["ID"], "Label": y_test_pred})
submission_df.to_csv("submission.csv", index=False)

print("Test set predictions saved to 'submission.csv'")

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_tfidf, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10)
)

# Compute mean and std of accuracy
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Compute ROC curve and AUC
y_val_prob = best_model.predict_proba(X_val_tfidf)[:, 1]  # Get probability estimates for positive class
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

plt.plot(train_sizes, test_mean, label="Validation Score", color="red", marker="s")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve for Logistic Regression")
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()