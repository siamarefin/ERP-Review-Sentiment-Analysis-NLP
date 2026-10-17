# ERP Review Sentiment Analysis: Machine Learning, Deep Learning & Transformers

**An experimental NLP project comparing classical machine learning, recurrent neural networks, and RoBERTa for three-class sentiment classification of Enterprise Resource Planning (ERP) software reviews.**

**Topics:** Natural Language Processing (NLP) · Sentiment Analysis · Text Classification · Transformer Fine-Tuning · GRU · LSTM · Model Evaluation

## Project Overview

ERP software reviews contain valuable feedback about usability, workflows, implementation, and product experience. This project explores how different NLP modeling approaches identify **positive**, **neutral**, and **negative** sentiment in user reviews. I compare TF–IDF-based classifiers, recurrent neural networks, and a fine-tuned RoBERTa transformer, using classification metrics and confusion matrices to examine both overall performance and minority-class behavior.

**Research questions**

1. How do conventional ML classifiers compare with GRU, LSTM, and a transformer model on ERP review sentiment?
2. How do the models handle a dataset dominated by positive reviews?
3. Does transformer fine-tuning improve macro-averaged performance, beyond overall accuracy?

## Dataset

The notebook loads a CSV named `erp_reviews_with_sentiment.csv` from a Kaggle input directory. The dataset's original download source and licensing are **not specified in the notebook**, so they should be documented before sharing or redistributing it.

| Property | Details |
|---|---|
| Total rows loaded | **2,950** |
| Text field | `Review_Answer` |
| Label field | `Sentiment_Label` |
| Sentiment categories | Negative, Neutral, Positive |
| Encoded labels | Negative = 0; Neutral = 1; Positive = 2 |
| Split for model experiments | 80% train / 20% test, `random_state=42` |
| Test-set size | **590** reviews |

**Original label counts** (shown in the modeling section): 2,100 Positive, 534 Neutral, and 316 Negative. The positive class is therefore approximately 71% of the data.

**EDA note:** A separate exploratory-analysis branch drops missing reviews and duplicate `(review, label)` pairs. That branch finds **one missing review** and **657 exact duplicate pairs**, leaving **2,292** records (1,611 Positive, 424 Neutral, 257 Negative). Later modeling cells **reload the original 2,950-row CSV** rather than train on this deduplicated EDA subset. The two counts are different by design of the notebook's execution flow and must not be confused.

## Workflow

```text
ERP reviews CSV
      |
      v
Exploratory analysis (missing values, duplicates, label balance, lengths)
      |
      v
Original dataset reloaded for modeling
      |
      v
Label encoding + 80/20 train/test split
      |
      +--------------------+------------------------+
      |                    |                        |
      v                    v                        v
TF–IDF features     Tokenized sequences      RoBERTa tokenization
      |              + padded inputs                |
      v                    |                        v
Logistic Regression    GRU and LSTM          RoBERTa fine-tuning
Random Forest             |                        |
      +--------------------+------------------------+
                           |
                           v
             Classification reports + confusion matrices
```

## Exploratory Data Analysis

The notebook includes:

- Class-frequency and class-proportion analysis.
- Missing-value and exact duplicate checks.
- Review-length distributions (characters and words), including class-wise summaries.
- Correlation-free text quality checks such as shortest/longest review inspection.
- Token and vocabulary exploration.

The notebook also contains an **NLTK cleaning experiment**: filtering very short reviews, removing punctuation/digits, removing stopwords, and applying lemmatization. These steps are explored in the notebook, but **the main model comparisons use the reloaded original text**, not necessarily that cleaned `final_text` field.

## Models and Methods

### 1. Classical Machine Learning

The notebook converts training reviews into **TF–IDF features** (`max_features=5000`) and evaluates:

- **Logistic Regression**
- **Random Forest Classifier**

The TF–IDF vectorizer is fit on training text and used to transform the test text. Model predictions are evaluated with a classification report and confusion matrix. The fitted classifiers are saved as `logistic_regression_model.pkl` and `random_forest_model.pkl`.

### 2. GRU Network

A Keras GRU text classifier uses:

- Tokenizer vocabulary limit: **6,376**
- Sequence length: **100** tokens, with padding
- Embedding dimension: **4**
- Stacked GRU layers: **128**, then **64** units
- Dropout: **0.5** after each recurrent layer
- Three-unit softmax output
- Adam optimizer and categorical cross-entropy loss
- Batch size: **64**; epochs: **10**
- Class weights computed from training labels

The model is saved as `gru_model.h5`.

### 3. LSTM Network

The LSTM experiment follows a similar Keras pipeline, replacing GRU layers with **128-unit and 64-unit LSTM layers**, using an **8-dimensional embedding** and dropout. It is trained for **10 epochs** with class weights and saved as `LSTM_model.h5`.

### 4. RoBERTa Transformer

The notebook fine-tunes `roberta-base` for three-class sequence classification using **PyTorch** and **Hugging Face Transformers**:

- Tokenization with truncation and padding, maximum length **128**
- `RobertaForSequenceClassification` with **3** labels
- AdamW optimizer, learning rate **5e-5**
- Batch size **16**
- **5** training epochs
- Cross-entropy-based classification objective
- GPU if available, otherwise CPU

The notebook then predicts labels for the test set and visualizes a confusion matrix. A later inference cell references `/kaggle/working/roberta_emotion_model`; **ensure the fine-tuned weights and tokenizer are actually saved to that directory** before using the loading example.

## Results

### Overall Comparison

The following values are reproduced from the **saved notebook outputs**, not from a new run.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 84% | 0.86 | 0.58 | 0.63 |
| Random Forest | 88% | 0.90 | 0.69 | 0.74 |
| GRU | 81.86% | 0.65 | 0.68 | 0.67 |
| LSTM | 86.78% | 0.75 | 0.74 | 0.74 |
| **RoBERTa** | **92%** | **0.86** | **0.83** | **0.84** |

*The notebook's scikit-learn classification reports round their displayed accuracies to two decimal places; GRU and LSTM test accuracies are taken from their explicit Keras evaluation outputs.*

### Per-Class F1 Scores

| Model | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Logistic Regression | 0.34 | 0.63 | 0.91 |
| Random Forest | 0.53 | 0.77 | 0.94 |
| GRU | 0.48 | 0.61 | 0.91 |
| LSTM | 0.56 | 0.71 | 0.94 |
| **RoBERTa** | **0.78** | **0.79** | **0.96** |

### Key Observations

1. **RoBERTa was the strongest evaluated model** in the saved run, with **92% test accuracy** and **0.84 macro F1**.
2. **Class imbalance matters.** The positive class dominates the dataset, so accuracy alone can obscure comparatively weak Negative and Neutral predictions.
3. **Classical models remain useful baselines.** Random Forest achieved approximately **88% accuracy**, outperforming the GRU and slightly exceeding the LSTM on overall accuracy in the recorded run.
4. **RoBERTa improved minority-class F1** compared with the other evaluated models, reaching **0.78 Negative F1** and **0.79 Neutral F1**.
5. The results are from the notebook's recorded train/test experiment; they do **not** establish robustness across multiple random splits or datasets.

## Technology Stack

- **Language:** Python
- **Data manipulation:** pandas, NumPy
- **EDA/visualization:** Matplotlib, Seaborn
- **Text processing:** NLTK, Keras tokenizer and sequence padding, scikit-learn TF–IDF
- **Classical ML:** scikit-learn
- **Deep learning:** TensorFlow / Keras (GRU and LSTM)
- **Transformers:** PyTorch, Hugging Face Transformers (`roberta-base`)
- **Evaluation:** scikit-learn classification reports and confusion matrices
- **Environment used in notebook:** Kaggle-style notebook environment

## Getting Started

### Requirements

Use a Python environment with packages compatible with your TensorFlow, PyTorch, and Transformers installations. Example installation:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk \
  tensorflow torch transformers tqdm notebook
```

Some NLTK operations require downloaded resources:

```python
import nltk
for resource in ('punkt', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(resource)
```

### Run the notebook

1. Clone the repository and enter its directory.
2. Obtain the **authorized ERP review CSV** and place it in an accessible location.
3. Update the notebook's hardcoded Kaggle input path:

   ```python
   data = pd.read_csv('/kaggle/input/erp-datasett/erp_reviews_with_sentiment.csv')
   ```

   For a local folder, for example, change it to:

   ```python
   data = pd.read_csv('data/erp_reviews_with_sentiment.csv')
   ```

4. Open `ERP_Sentiment_Analysis_ML_DL_Transformers.ipynb` in Jupyter or Kaggle and run the cells in order.
5. Review the evaluation reports and confusion matrices for each model.

**Important:** This notebook is a research experiment, not a plug-and-play command-line application. The CSV file is not included in this README, and the notebook contains environment-specific paths and model-saving/loading steps that should be checked before claiming full reproducibility.

## Suggested Repository Layout

This is a **recommended organization**, not a claim that all these files already exist:

```text
ERP-Review-Sentiment-Analysis-NLP/
├── README.md
├── ERP_Sentiment_Analysis_ML_DL_Transformers.ipynb
├── data/
│   └── README.md              # Dataset source, access, and license details
├── results/                   # Exported plots and metric summaries (optional)
└── requirements.txt           # Freeze verified package versions (recommended)
```

Avoid uploading proprietary or unlicensed review datasets and large model checkpoint files without confirming redistribution rights.

## Limitations and Future Work

- **Class imbalance:** Positive reviews are the majority class; consider macro F1, per-class recall, and more robust imbalance mitigation.
- **Duplicate handling:** EDA identifies duplicate pairs, but the modeling dataset is reloaded from the original file. Deduplicate **before splitting** in a future run to test for possible train/test overlap.
- **Validation design:** The notebook passes the test set as `validation_data` during GRU/LSTM training. A dedicated validation partition should be used during development, reserving the test set for final evaluation.
- **Reproducibility:** Evaluate repeated random seeds and report variation in scores; record library versions and full training settings.
- **Domain generalization:** Assess performance on other ERP platforms or separate datasets to test robustness beyond the current review collection.
- **Comparative transformer analysis:** Explore additional pretrained models, training strategies, computational costs, and error analysis for minority classes.
- **Deployment:** Package the best-performing model into a documented inference pipeline only after validating preprocessing and the saved tokenizer/model artifacts.

## Research Takeaway

This project demonstrates an end-to-end experimental comparison of **TF–IDF classifiers, recurrent neural networks, and a fine-tuned transformer** for ERP review sentiment analysis. In the recorded experiment, **RoBERTa produced the best overall accuracy and macro F1**, including stronger performance on the minority Negative and Neutral classes. The analysis highlights the importance of class-wise evaluation and careful experimental design in applied NLP.

---

**Keywords:** `NLP` `ERP Reviews` `Sentiment Analysis` `Text Classification` `RoBERTa` `Transformers` `GRU` `LSTM` `Random Forest` `TF-IDF` `Machine Learning`
