# Financial Sentiment Analysis with Traditional & LLM Models

This project evaluates multiple approaches to sentiment classification on financial text data — ranging from traditional ML models to zero-shot large language models (LLMs).

---

## Overview

We benchmark the performance of several models on **financial sentiment classification**, comparing accuracy, F1 scores, and per-source breakdowns across diverse datasets.

### Models Compared

| Model               | Type           | Notes                           |
| ------------------- | -------------- | ------------------------------- |
| Logistic Regression | Traditional ML | TF-IDF + LogisticRegression     |
| FinBERT             | Transformer    | Finetuned BERT on finance texts |
| SpaCy + TextBlob    | Rule-based     | Polarity → Label mapping        |
| o4-mini (Azure)     | LLM            | o4-mini via Azure OpenAI        |

---

## Folder Structure

```
project-root/
├── notebooks/               ← Main experiment notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_logistic_regression_baseline.ipynb
│   ├── 3_finbert_inference.ipynb
│   ├── 4_spacy_textblob.ipynb
│   ├── 5_o4mini_inference.ipynb
│   └── demo.ipynb          ← 📊 Summary notebook with visuals
│
├── scripts/                ← Reusable utilities
│   ├── metrics.py
│   ├── preprocessing.py
│   └── plot_utils.py
│
├── data/
│   └── processed/          ← Cleaned CSVs from preprocessing
│
├── models/                 ← Saved predictions & classification reports
│
└── README.md
```

---

## 📦 Setup

```bash
git clone https://github.com/your-username/financial-sentiment-analysis.git
cd financial-sentiment-analysis
pip install -r requirements.txt
```

> Requires: Python 3.10+, Jupyter, scikit-learn, transformers, spacy, seaborn

Optional:

```bash
python -m spacy download en_core_web_sm
```

---

## Run Notebooks

Start by running the following notebooks in order:

1. `1_data_preprocessing.ipynb` – Loads and prepares datasets
2. `2_logistic_regression_baseline.ipynb`
3. `3_finbert_inference.ipynb`
4. `4_spacy_textblob.ipynb`
5. `5_o4mini_inference.ipynb`
6. `demo.ipynb` – 📊 Visit this first! Summary visualizations and model comparison.

---

## 📊 Demo Preview

<!-- <img src="https://user-images.githubusercontent.com/demo-placeholder.png" width="600"> -->

> A side-by-side comparison of model F1 scores by source and sentiment class is provided in `demo.ipynb`.

---

## Notes on Azure OpenAI

To run `5_o4mini_inference.ipynb`, you need:

-   Azure OpenAI access
-   A deployed `o4-mini` model (deployment name: `o4-mini`)
-   Your endpoint and key in the notebook (suggest using environment variables)

---

## 📜 License

This project is for educational and research purposes only. You may adapt and share with attribution.

---

## Acknowledgements

-   [FiQa 2018](https://huggingface.co/datasets/pauri32/fiqa-2018)
-   [FinBERT](https://huggingface.co/ProsusAI/finbert)
-   [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
-   [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
