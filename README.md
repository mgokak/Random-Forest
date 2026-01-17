# Random Forest – Classification & Regression

## Overview

This repository contains Jupyter Notebooks that demonstrate **Random Forest models** for both **classification** and **regression** tasks. The notebooks focus on understanding how ensemble learning works, how multiple decision trees are combined, and how Random Forest improves performance and stability compared to a single decision tree.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Random Forest Classification  
4. Random Forest Regression  
5. Model Evaluation 

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `RandomForestClassification.ipynb` – Random Forest applied to a classification problem  
- `RandomForestRegression.ipynb` – Random Forest applied to a regression problem  

---

## Random Forest Classification

### `RandomForestClassification.ipynb`

This notebook applies **Random Forest Classifier** on a real-world classification dataset.

Key points:
- Uses multiple decision trees to improve prediction accuracy
- Reduces overfitting compared to a single decision tree
- Works well with non-linear data and mixed feature types

Basic commands used:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

The notebook also includes dataset exploration and preprocessing before training the model.

---

## Random Forest Regression

### `RandomForestRegression.ipynb`

This notebook demonstrates **Random Forest Regression** for predicting continuous target values.

Key points:
- Uses an ensemble of decision trees for regression
- Captures complex non-linear relationships
- Robust to outliers and noise

Basic commands used:
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
```

The notebook focuses on data preparation and evaluating regression performance.

---

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
