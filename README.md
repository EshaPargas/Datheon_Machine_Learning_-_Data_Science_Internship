# Datheon Machine Learning & Data Science Internship

This repository contains my completed tasks for the Datheon Machine Learning & Data Science Internship.

## Repository structure

```
.
├── README.md
├── requirements.txt
└── task_01_multi_class_classification/
    └── iris_classification_pipeline.ipynb
└── task_02_house_price_regression/
    └── house_price_regression.ipynb
```

> Each additional completed task goes in its own folder (`task_XX_name/`) with its notebook or source code, and gets its own section below.

## Setup

```bash
git clone https://github.com/EshaPargas/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship
pip install -r requirements.txt
jupyter notebook
```

## Tasks

### Multi-Class Benchmark Classification Pipeline (Iris)

**Folder:** `task_01_multi_class_classification/`

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


### California House Price Prediction: OLS vs. Ridge vs. Lasso

**Folder:** `task_02_house_price_regression/`

Predicts median house values on the California Housing dataset and compares three linear models.

**What it does**

- Loads the data with Pandas, imputes missing total_bedrooms values (median), and plots a correlation heatmap
- Engineers ratio features (rooms, bedrooms and population per household) and interaction features (income x rooms per household, income x house age, latitude x longitude)
- Standardizes numeric features and one-hot encodes ocean_proximity, all inside a Scikit-Learn Pipeline so there is no data leakage
- Diagnoses multicollinearity with Variance Inflation Factors (VIF)
- Fits OLS, plus Ridge (L2) and Lasso (L1) tuned with GridSearchCV (5-fold CV over a wide alpha range)
- Benchmarks RMSE, MAE and R² on a held-out 20% test set
- Plots predicted vs. actual values, residual distributions, residuals vs. predicted, and a coefficient comparison showing which features Lasso removes

**Results: the notebook prints the full benchmark table (RMSE, MAE, R²) for all three models when run.**
