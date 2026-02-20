# Twitter Sentiment Analysis Evolution

This repository showcases the progression of Natural Language Processing (NLP) techniques applied to the same English-language Twitter dataset. The project is divided into three distinct stages, reflecting the shift from frequency-based machine learning to context-aware transformer models.

This series of projects was developed as part of the **Deep Learning for NLP** course during the **Spring Semester of 2025**.

## 📊 Project Overview

The goal of this series was to develop a sentiment classifier (Binary: Positive/Negative) and observe how different architectures and text representations impact model performance.

### 🔍 Stage 1: Traditional Machine Learning
*   **Model:** Logistic Regression
*   **Feature Extraction:** TF-IDF (Unigrams, Bigrams, and Trigrams)
*   **Highlights:** Extensive text preprocessing, vocabulary cleaning, and n-gram analysis.
*   **Result:** Established a strong baseline by leveraging word frequencies.

### 🧠 Stage 2: Deep Learning with Static Embeddings
*   **Model:** Deep Neural Network (DNN) / Multi-Layer Perceptron
*   **Feature Extraction:** Pre-trained GloVe Embeddings (`glove-twitter-200`)
*   **Library:** PyTorch
*   **Highlights:** Implementation of Batch Normalization, Dropout for regularization, and Hyperparameter optimization using Optuna.
*   **Result:** Explored the limitations of averaged static word vectors compared to frequency-based models.

### 🤖 Stage 3: Transformer Models (SOTA)
*   **Model:** BERT (base-uncased) & DistilBERT
*   **Technique:** Fine-tuning pre-trained Transformers
*   **Library:** Hugging Face Transformers + PyTorch
*   **Highlights:** Leveraged context-aware embeddings and the self-attention mechanism. Optimized with AdamW and linear learning rate scheduling.
*   **Result:** Achieved peak accuracy (~85.6%), demonstrating the power of transfer learning in NLP.

---

## 📈 Performance Comparison

| Model | Approach | Feature Extraction | Accuracy (Val) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Traditional ML | TF-IDF | ~80.5% |
| **DNN** | Deep Learning | GloVe Twitter | ~79.2% |
| **DistilBERT** | Transformer | Contextual | ~84.8% |
| **BERT** | Transformer | Contextual | **85.6%** |

---