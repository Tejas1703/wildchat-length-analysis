# wildchat-length-analysis
A reproducible machine learning pipeline that studies whether the **first user message** in an LLM conversation can predict how the conversation will evolve.

The project explores three core tasks:
1. Predicting conversation length
2. Predicting user intent
3. Understanding how intent influences conversation duration

---

## Overview

Modern LLM chat systems generate millions of conversations daily.  
Understanding early signals in a conversation can help with:

- Workload forecasting
- System design
- User behavior analysis
- Safety and moderation planning

This repository investigates how much information is contained in the **very first user prompt**.

---

## Research Questions

### RQ1 — Conversation Length Prediction
Can the first user message predict whether a conversation becomes:

- Short
- Medium
- Long

This is framed as a 3-class text classification problem.

---

### RQ2 — Intent Prediction
Can the first user message predict the user's primary intent?

Heuristic labels:
- Question Answering
- Coding
- Writing

---

###  Intent vs Conversation Length
How strongly does predicted intent correlate with conversation duration?

This analysis provides behavioral insights into how users interact with LLMs.

---

## Dataset

This project uses the **WildChat dataset** and constructs a conversation-level dataset where each conversation contributes:

- First user message text
- Conversation length bucket
- Heuristic intent label

Large raw and intermediate artifacts are excluded from the repository.

---

## Repository Structure

```text
wildchat-length-analysis/
│
├── notebooks/
│ └── WildChat_Analysis_Pipeline.ipynb
│
├── outputs/
│ ├── figures/
│ │ ├── intent_distribution.png
│ │ ├── intent_vs_turnbucket_heatmap.png
│ │ ├── p_long_given_intent.png
│ │ └── rq1_confusion_best_test.png
│ │
│ └── tables/
│ ├── Research_Question_1_Result_Table.jpeg
│ └── Research_Question_2_Results_Table.jpeg
│
└── README.md
```

---

## Methodology

### Text Representation
First user messages are converted to numerical features using **TF-IDF vectorization**.

### Models Evaluated
The following lightweight text classifiers were evaluated:

- Logistic Regression  
- Linear Support Vector Machine (SVM)  
- Multinomial Naive Bayes  

Evaluation strategy:
- Train / Validation / Test split  
- Cross-validation  
- Macro F1 score as the primary metric  

---

## Key Findings

### Predicting Conversation Length
The first user message contains strong signals about how long a conversation will continue.

Confusion matrix: outputs/figures/rq1_confusion_best_test.png

---

### Predicting User Intent
Lightweight models can reliably distinguish between:
- QA prompts
- Coding prompts
- Writing prompts

---

### Intent vs Conversation Duration

A clear behavioral pattern emerges:

- Coding prompts are most likely to lead to long conversations
- Writing prompts also frequently produce long interactions
- QA prompts tend to result in shorter conversations

Visualization: outputs/figures/p_long_given_intent.png

---

## Reproducing the Pipeline

### Clone repository
git clone https://github.com/Tejas1703/wildchat-length-analysis.git

cd wildchat-length-analysis


### Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

### Run the notebook
Open and execute:
notebooks/WildChat_Analysis_Pipeline.ipynb

Running the notebook will regenerate all figures and tables.

---

## Notes on Data Availability

Large artifacts are intentionally excluded:
- Raw dataset
- Processed datasets
- Trained models

This keeps the repository lightweight while preserving reproducibility.

---

## Author

Tejeshwar Sarma
