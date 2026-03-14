# 🏠 House Prices Prediction
### Kaggle Midterm Project — Introduction to Machine Learning

Predicting house sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) with Python and scikit-learn.

---

## About the Project

This project walks through a full machine learning pipeline — from data cleaning and feature engineering to training and comparing three different models. The goal is to predict the sale price of a house based on features like size, age, quality, and number of bathrooms.

---

## Dataset

- **Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **File used:** `train.csv` only
- **Size:** ~1,460 rows × 79 features
- **Target:** `SalePrice` (log-transformed during training)

---

## What's in the Notebook

1. Exploratory Data Analysis
2. Missing value imputation
3. Outlier removal
4. Feature engineering (5 new features)
5. Feature selection using Random Forest importance
6. Model training and evaluation
7. Model comparison

---

## New Features Created

| Feature | Description |
|---|---|
| `TotalSF` | Combined basement + 1st floor + 2nd floor area |
| `TotalBathrooms` | Weighted sum of all bathroom columns |
| `HouseAge` | Years since the house was built |
| `OverallScore` | Overall quality × overall condition |
| `HasGarage` / `HasPool` / `Has2ndFloor` | Binary yes/no flags |

---

## Model Results

| Model | RMSE (log) | R² |
|---|---|---|
| Linear Regression | — | — |
| KNN | — | — |
| Random Forest | — | — |

> Fill in your actual scores after running the notebook

**Best model:** Random Forest

---

## How to Run

1. Download `train.csv` from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Place it in the same folder as the notebook
3. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
4. Open and run `house_prices_midterm.ipynb`

---

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualizations
- `scikit-learn` — machine learning models
