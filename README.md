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

> Screenshots/figures you paste in the report are saved under `artifacts_legal_similarity/figs/`.

---

## 3. Dataset
- Kaggle: **Legal Clause Dataset** by *Bahushruth*  
  https://www.kaggle.com/datasets/bahushruth/legalclausedataset

The notebook uses **`kagglehub`** to download automatically in Colab.

---

## 4. Quick Start (Colab)

Open the notebook in Colab and run cells in order. Minimal snippet:

```python
!pip install kagglehub --quiet
import kagglehub, os, pandas as pd

DATA_DIR = kagglehub.dataset_download("bahushruth/legalclausedataset")
print("Dataset at:", DATA_DIR)

# ...then follow cells to:
# - load & clean all CSVs
# - build balanced pairs (same category = positive, different = negative)
# - tokenize & pad (vocab=20k, maxlen=120)
# - stratified train/val split
# - train Model A & Model B
# - evaluate + save plots
