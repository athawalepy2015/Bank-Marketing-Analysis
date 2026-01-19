# Bank Marketing Analysis Using Machine Learning

An end-to-end machine learning project that analyzes real-world bank telemarketing data to predict whether a customer will subscribe to a term deposit. The project focuses on data preprocessing, handling class imbalance, model comparison using appropriate metrics, and translating results into actionable business insights.

---

## Project Objective

The goal of this project is to:
- Understand customer behavior in bank telemarketing campaigns
- Predict whether a customer will subscribe to a term deposit (`yes` / `no`)
- Identify key factors that influence subscription decisions
- Support data-driven marketing strategies such as customer targeting and campaign timing

This project treats the problem as a **binary classification task** with real-world constraints such as class imbalance and noisy customer data.

---

## Dataset

- **Source:** UCI Machine Learning Repository — Bank Marketing Dataset  
- **File used:** `bank-additional-full.csv`  
- **Original size:** 41,188 rows × 21 columns  
- **After cleaning:** 30,439 rows × 32 columns (after encoding)

### Feature Categories
- **Customer attributes:** age, job, marital status, education, loans
- **Contact details:** contact type, month, weekday, call duration
- **Campaign history:** number of contacts, days since last contact, previous outcomes
- **Economic indicators:** employment variation, euribor rate, consumer confidence
- **Target:** `y` → subscription outcome (encoded as `y_enc`)

---

## Data Preprocessing

Key preprocessing steps:
- Replaced `"unknown"` values with `NaN` and removed incomplete rows
- Removed unrealistic outliers:
  - Call duration > 2000 seconds
  - Age outside the range 18–100
- Encoded categorical variables using `LabelEncoder`
- Scaled numerical features using `StandardScaler`
- Performed **stratified train–test split** to preserve class distribution
- Addressed class imbalance using **SMOTE** on the training set only

> ⚠️ **Data Leakage Note**  
> The feature `duration` (call length) is known only *after* a call occurs. Including it improves predictive performance but introduces information leakage for pre-call targeting.  
> In this project, `duration` is included to analyze **post-call subscription behavior**. For pre-call customer targeting, this feature should be excluded.

---

## Exploratory Data Analysis (EDA)

Key insights from visualization and aggregation:
- Certain occupations (e.g., students, retired customers) show higher subscription rates
- Subscription likelihood varies significantly by month, showing seasonal patterns
- Longer call duration strongly correlates with subscription outcome
- Age shows a non-linear relationship with subscription probability
- Economic indicators contribute meaningfully to prediction performance

---

## Machine Learning Models

We trained and evaluated multiple supervised learning models to understand tradeoffs between interpretability, performance, and robustness.

### Models Trained
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) *(trained on a subset due to computational cost)*
- Neural Network (MLPClassifier)

### Evaluation Metrics
Given class imbalance, models were evaluated using:
- Accuracy
- Precision / Recall / F1-score
- ROC–AUC
- Average Precision (PR–AUC)
- Confusion Matrix
- ROC and Precision–Recall curves

---

## Model Performance Summary

| Model                | Accuracy | AUC   | Avg Precision |
|---------------------|----------|-------|---------------|
| Logistic Regression | 0.849    | 0.930 | 0.590         |
| **Random Forest**   | **0.898**| **0.940** | **0.647** |
| KNN                 | 0.840    | 0.866 | 0.436         |
| SVM                 | 0.901    | 0.911 | 0.637         |
| Neural Network      | 0.883    | 0.911 | 0.558         |

### Best Model: Random Forest
Random Forest was selected as the best overall model because it provided the strongest balance between precision, recall, and AUC — not just accuracy. It also offered feature importance scores, improving interpretability.

Top influential features included:
- `duration`
- `euribor3m`
- `nr.employed`
- `pdays`
- campaign-related features

---

## Feature Importance

Feature importance from the Random Forest model highlights that both **customer engagement metrics** (call duration, previous contacts) and **economic indicators** play a significant role in subscription decisions.

---

## Curve Fitting: Age vs Subscription Rate

A third-degree polynomial regression was used to model the non-linear relationship between age and subscription rate:
- **RMSE:** 0.1349  
- **R²:** 0.6151  

Results show increased subscription likelihood among older customers, especially after age 60.

---

## Unsupervised Learning: Customer Segmentation

To explore customer segments:
- **PCA** was used for dimensionality reduction and visualization
- **K-Means clustering (k = 5)** identified distinct customer behavior groups

Although the first two PCA components explain ~29% of the variance, the visualization reveals meaningful customer segmentation patterns useful for marketing strategy.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Google Colab / Jupyter Notebook

---
