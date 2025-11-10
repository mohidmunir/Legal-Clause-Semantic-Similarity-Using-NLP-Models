# Legal-Clause-Semantic-Similarity-Using-NLP-Models
# LexiSim
Legal Clause Semantic Similarity — **non-transformer baselines (Siamese BiLSTM & BiLSTM+Attention)** for CS-452 Assignment 02.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)]()
[![Colab-Friendly](https://img.shields.io/badge/Run-Colab-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> Predict whether two legal clauses convey the **same** or **different** meaning without using pretrained transformers.  
> Dataset: Kaggle **Legal Clause Dataset** (395 CSVs, 150k+ clauses).

---

## 1. Project Overview
This repo implements two baseline architectures for **legal clause semantic similarity**:

- **Model A – Siamese BiLSTM** (shared embedding + BiLSTM; |h1−h2| and h1⊙h2 → MLP)
- **Model B – BiLSTM + Additive Attention** (encode each clause with attention; compare encodings)

Both are trained **from scratch** (no BERT/roberta/legal-BERT).  
Metrics: **Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC**.

---

## 2. Repo Structure
