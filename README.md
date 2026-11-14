# Datheon Machine Learning & Data Science Internship

This repository contains my completed tasks for the Datheon Machine Learning & Data Science Internship.

## Repository structure

```
.
├── README.md
├── requirements.txt
└── task_02_multi_class_classification/
    └── iris_classification_pipeline.ipynb
```

> Each additional completed task goes in its own folder (`task_XX_name/`) with its notebook or source code, and gets its own section below.

## Setup

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
pip install -r requirements.txt
jupyter notebook
```

## Tasks

### Multi-Class Benchmark Classification Pipeline (Iris)

**Folder:** `task_02_multi_class_classification/`

An end-to-end supervised classification pipeline on the Iris dataset using Python, Pandas, Scikit-Learn, Seaborn and Matplotlib.

**What it does**
- Loads the Iris dataset with Pandas and explores it with a Seaborn pairplot and correlation heatmap
- Splits the data 80/20 (stratified) and scales features with `StandardScaler`, fitted on the training set only to avoid data leakage
- Trains `LogisticRegression` and `DecisionTreeClassifier`
- Benchmarks accuracy, precision, recall and F1-score with full classification reports
- Plots confusion matrix heatmaps and decision-boundary visualizations (petal length vs. petal width)
- Adds a stratified 5-fold cross-validation check for a more stable comparison

**Results (30-sample held-out test set)**

| Model | Accuracy | Macro F1 | 5-fold CV accuracy |
|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.953 |
| Decision Tree | 0.967 | 0.967 | 0.960 |

Setosa is separated perfectly by both models; all errors are between versicolor and virginica.

<!-- Add a section here for each additional task you complete. -->
