"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenization
    tokenized = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set(word for doc in tokenized for word in doc)

    # 3. Priors
    priors = {
        1: np.mean(labels == 1),
        0: np.mean(labels == 0)
    }

    # 4. Word counts per class
    word_counts = {
        1: {word: 0 for word in vocab},
        0: {word: 0 for word in vocab}
    }

    total_words = {1: 0, 0: 0}

    for doc, label in zip(tokenized, labels):
        for word in doc:
            word_counts[label][word] += 1
            total_words[label] += 1

    # 5. Word probabilities (MLE)
    word_probs = {
        1: {},
        0: {}
    }

    for c in [0, 1]:
        for word in vocab:
            if total_words[c] > 0:
                word_probs[c][word] = word_counts[c][word] / total_words[c]
            else:
                word_probs[c][word] = 0

    # 6. Prediction
    test_tokens = test_email.split()

    scores = {}
    for c in [0, 1]:
        score = np.log(priors[c])
        for word in test_tokens:
            if word in vocab and word_probs[c][word] > 0:
                score += np.log(word_probs[c][word])
            else:
                score += np.log(1e-9)  # avoid log(0)
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. Prediction function
    def predict(X_train, y_train, x_test, k):
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(x_test, X_train[i])
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        labels = [label for _, label in neighbors]
        return max(set(labels), key=labels.count)

    # 5. Predictions
    train_preds = np.array([
        predict(X_train, y_train, x, k) for x in X_train
    ])

    test_preds = np.array([
        predict(X_train, y_train, x, k) for x in X_test
    ])

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds