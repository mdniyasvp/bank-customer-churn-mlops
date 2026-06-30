# Bank Customer Churn Prediction - Model Selection Report

## Project Overview

This project aims to predict whether a bank customer is likely to churn using supervised machine learning. Multiple classification algorithms were evaluated using a consistent preprocessing pipeline, identical train-test split, and standardized evaluation metrics.

The objective was not simply to achieve the highest accuracy, but to identify the most suitable production model by considering predictive performance, robustness, training efficiency, and reproducibility.

---

# Dataset Summary

* **Dataset:** Bank Customer Churn
* **Records:** 165,034
* **Features:** 14
* **Target Variable:** Exited
* **Problem Type:** Binary Classification

Target Distribution

| Class       | Percentage |
| ----------- | ---------: |
| Stayed (0)  |     78.84% |
| Churned (1) |     21.16% |

The dataset is moderately imbalanced; therefore, evaluation focused on Precision, Recall, F1-score, and ROC-AUC rather than relying solely on Accuracy.

---

# Feature Engineering

The following preprocessing pipeline was applied consistently to every model:

### Dropped Features

* id
* CustomerId
* Surname

### Numerical Features

* CreditScore
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

### Categorical Features

* Geography
* Gender

### Preprocessing

* One-Hot Encoding for categorical variables
* Standard Scaling for numerical variables
* Unified Scikit-learn Pipeline

This ensured identical preprocessing across all experiments.

---

# Models Evaluated

## 1. Logistic Regression

Purpose:

Establish a simple and interpretable baseline model.

Performance:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8334 |
| Precision | 0.6933 |
| Recall    | 0.3813 |
| F1 Score  | 0.4920 |
| ROC-AUC   | 0.8145 |

Observations

* Fast to train
* Highly interpretable
* Limited ability to model nonlinear relationships
* Lowest Recall among all evaluated models

---

## 2. Random Forest

Purpose:

Capture nonlinear feature interactions and improve predictive performance.

Performance:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8584 |
| Precision | 0.7238 |
| Recall    | 0.5349 |
| F1 Score  | 0.6152 |
| ROC-AUC   | 0.8740 |

Observations

* Significant improvement over Logistic Regression
* Better balance between Precision and Recall
* More robust on structured tabular data

---

## 3. Tuned Random Forest

Hyperparameter optimization was performed using GridSearchCV with 5-fold cross-validation.

Best Parameters

* n_estimators = 200
* max_depth = 20
* min_samples_split = 2
* min_samples_leaf = 2

Performance

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8611 |
| Precision | 0.7358 |
| Recall    | 0.5361 |
| F1 Score  | 0.6203 |
| ROC-AUC   | 0.8812 |

Observations

* Improved generalization
* Reduced overfitting
* Better ROC-AUC compared to the baseline Random Forest

---

## 4. XGBoost

Purpose:

Evaluate a gradient boosting algorithm optimized for structured tabular datasets.

Performance

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8649 |
| Precision | 0.7405 |
| Recall    | 0.5566 |
| F1 Score  | 0.6355 |
| ROC-AUC   | 0.8886 |

Observations

* Best overall predictive performance
* Highest ROC-AUC
* Highest Recall
* Highest F1-score
* Fast training on the current dataset

---

# Model Comparison

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.8334 |     0.6933 |     0.3813 |     0.4920 |     0.8145 |
| Random Forest       |     0.8584 |     0.7238 |     0.5349 |     0.6152 |     0.8740 |
| Tuned Random Forest |     0.8611 |     0.7358 |     0.5361 |     0.6203 |     0.8812 |
| XGBoost             | **0.8649** | **0.7405** | **0.5566** | **0.6355** | **0.8886** |

---

# Final Model Selection

The XGBoost model was selected as the production candidate based on the following criteria:

* Highest Accuracy
* Highest Precision
* Highest Recall
* Highest F1-score
* Highest ROC-AUC
* Stable training performance
* Excellent suitability for structured tabular data

Although Logistic Regression offers superior interpretability and Random Forest provides competitive performance, XGBoost demonstrated the best overall balance between predictive accuracy and business value.

---

# MLOps Workflow

The project follows an end-to-end MLOps pipeline:

1. Data Versioning using DVC
2. Exploratory Data Analysis
3. Feature Engineering
4. Scikit-learn Pipeline
5. Multiple Model Training
6. Hyperparameter Tuning
7. MLflow Experiment Tracking
8. Production Model Selection
9. FastAPI Deployment (Next Phase)
10. Docker Containerization (Next Phase)
11. CI/CD Automation (Next Phase)
12. Cloud Deployment (Next Phase)

---

# Conclusion

Four machine learning models were evaluated using a consistent preprocessing pipeline and standardized evaluation metrics.

After systematic experimentation and hyperparameter tuning, XGBoost achieved the strongest overall performance and was selected as the production candidate for deployment.

The entire experimentation process was tracked using MLflow, ensuring reproducibility and enabling comparison across model versions. This workflow demonstrates an end-to-end MLOps approach, from data versioning and model development to experiment management and production readiness.
 
