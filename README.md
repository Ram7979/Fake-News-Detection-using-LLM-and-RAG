# Fake News Detection Using LLM and Retrieval-Augmented Generation (RAG)

## Project Overview

The rapid growth of social media and digital news platforms has significantly accelerated the spread of **fake news**, leading to misinformation, public distrust, and social instability. Traditional machine learning (ML) and deep learning (DL) models are effective at identifying **linguistic and stylistic patterns** of deception, but they **fail to verify factual correctness** against real-world evidence.

This project presents a **Hybrid Fake News Detection Framework** that combines:

* **Traditional Machine Learning classifiers** for fast and reliable pattern-based detection, and
* **Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG)** for real-time fact verification using external web evidence.

The system not only classifies news as *Fake* or *Real*, but also provides:

* Evidence-backed verification
* Human-readable explanations
* Transparent and interpretable decision-making

This work is based on the research paper **“Fake News Detection Using LLM and RAG”** .

---

## Key Objectives

* Design a **scalable and interpretable** fake news detection system
* Combine **statistical learning** with **semantic reasoning**
* Reduce LLM hallucinations using **retrieval grounding**
* Improve **accuracy, transparency, and trustworthiness** in misinformation detection

---

## Core Idea

Fake news detection should not be treated as a simple binary classification problem.

> **This project transforms fake news detection into an evidence-based reasoning process.**

The framework follows a **two-layer architecture**:

1. **Machine Learning Layer** – Detects deceptive linguistic patterns
2. **LLM + RAG Verification Layer** – Verifies claims using real-world evidence

---

## System Architecture

### High-Level Pipeline

1. Data Acquisition & Preprocessing
2. Feature Representation & Extraction
3. Machine Learning Classification
4. Retrieval-Augmented Verification (RAV)
5. Decision Fusion & Explanation Generation

---

### Architecture Explanation

#### 1)Data Acquisition & Preprocessing

* Noise removal (HTML tags, punctuation, stopwords)
* Text normalization (lowercasing, lemmatization)
* Tokenization
* Class balancing (equal fake and real samples)
* Headline + article body fusion

#### 2)Feature Representation

* TF-IDF vectorization (50,000 features)
* Unigrams and bigrams
* Optional word embeddings (Word2Vec / GloVe)

#### 3)Machine Learning Classification

Models used:

* Logistic Regression
* Passive-Aggressive Classifier
* (Baseline comparison with LSTM)

Output:

* Binary label: **Fake / Real**
* Confidence score

#### 4)Retrieval-Augmented Verification (RAV)

* Claim extraction using NLP
* Web search using **Serper Search API**
* Retrieval of relevant evidence (title, snippet, URL)
* Evidence passed to **LLM (GPT-4)** using **LangChain**

LLM Verdict:

* **Supported**
* **Refuted**
* **Inconclusive**

Includes a clear explanation with citations.

#### 5)Decision Fusion & Explainability

* ML confidence + LLM verdict are combined
* Final decision is generated with:

  * Prediction
  * Explanation
  * Evidence sources

---

## Dataset & Experimental Setup

* **Dataset**: Kaggle Fake News Dataset
* **Total Samples**: ~20,800
* **Train-Test Split**: 80:20 (stratified)
* **Feature Extraction**: TF-IDF (50,000 features)
* **Evaluation Metrics**:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

---

## Results & Performance

### Model Performance Comparison

| Model                     | Accuracy (%) |
| ------------------------- | ------------ |
| Naïve Bayes               | 90.1         |
| Logistic Regression       | 94.2         |
| Passive-Aggressive        | 93.6         |
| LSTM (Deep Learning)      | 95.4         |
| **Hybrid ML + LLM + RAG** | **96.8**     |

### Impact of RAG on LLM Accuracy

| LLM        | Without RAG (%) | With RAG (%) |
| ---------- | --------------- | ------------ |
| GPT-3.5    | 93.2            | 95.6         |
| GPT-4      | 94.3            | **96.8**     |
| Gemini 1.5 | 93.9            | 95.9         |
| Claude 3   | 94.1            | 96.1         |
| LLaMA-3    | 92.6            | 94.8         |

RAG significantly improves factual accuracy and reduces hallucinations.

---

## Project Structure

```
fake-news-detection/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_ml.py
│   ├── evaluate.py
│   ├── inference.py
│   └── rav/
│       ├── claim_extraction.py
│       ├── retriever.py
│       └── llm_verifier.py
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── classifier.pkl
│
└── docs/
    └── research_paper.pdf
```

---

## Installation

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
```

---

## Usage

### Train ML Model

```bash
python src/train_ml.py
```

### Run Fake News Verification

```bash
python src/inference.py --text "Your news article here"
```

### Output Example

```json
{
  "ml_prediction": "Fake",
  "confidence": 0.87,
  "llm_verdict": "Refuted",
  "explanation": "The claim contradicts verified sources...",
  "sources": ["https://example.com"]
}
```

---

## Why This Project Matters

* Moves beyond **style-based detection**
* Introduces **real-world fact verification**
* Produces **explainable AI outputs**
* Aligns with **Responsible AI & XAI principles**
* Suitable for real-world deployment in:

  * Politics
  * Finance
  * Healthcare
  * Journalism

---

## Research Foundation

This project is grounded in:

* Statistical Learning Theory
* Information Retrieval Theory
* Transformer Attention Theory
* Ensemble Decision Fusion Theory

---

## Research Paper

*Fake News Detection Using LLM and RAG*
(See `/docs/research_paper.pdf`) 

---


## Acknowledgements

* OpenAI (GPT Models)
* LangChain
* Serper Web Search API
* Kaggle Fake News Dataset
