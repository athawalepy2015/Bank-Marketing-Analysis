# Bank Marketing Analysis Using Machine Learning

Predicting whether a customer will subscribe to a term deposit using real telemarketing campaign data from a Portuguese bank (2008–2010). This project combines EDA, supervised ML modeling, and customer segmentation to help identify the right target audience and improve campaign strategy.

---

## Project Goal

The goal of this project is to:
- Understand what factors influence term-deposit subscription (`y = yes/no`)
- Build ML models that can predict subscription likelihood for new customers
- Provide insights that help marketing/sales teams plan better campaigns (who to call, when to call, and what patterns matter most)

---

## Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing Dataset (Moro et al., 2014)  
**File used:** `bank-additional-full.csv`  
**Original size:** 41,188 rows × 21 columns  
**After cleaning:** ~30,439 rows (after removing missing/unknown values and unrealistic outliers)

### Feature Types
- **Customer details:** age, job, marital status, education, loans, housing, etc.
- **Contact details:** contact method, month/day, duration
- **Campaign history:** number of contacts, previous outcomes, recency of last contact
- **Economic indicators:** euribor rate, employment variation, consumer confidence, etc.
- **Target:** `y` (subscribed to term deposit)

---

## Key Questions Explored

This project investigates:
- Which customer groups subscribe the most?
- How does call behavior (duration, frequency) affect outcomes?
- Which months/periods perform best for marketing campaigns?
- How do past campaign outcomes affect future subscriptions?
- What features contribute most to predicting subscription?
- Can we discover distinct customer segments?

---

## Workflow

### 1) Data Cleaning & Preprocessing
- Replaced `"unknown"` with `NaN` and removed missing rows (dataset was large enough to support this)
- Filtered unrealistic values:
  - Age kept between **18 and 100**
  - Very long call durations removed (extreme outliers)
- Encoded categorical variables using `LabelEncoder`
- Scaled numeric variables using `StandardScaler`
- Train/test split with **stratification**

### 2) Handling Class Imbalance
The dataset contains significantly more `no` than `yes`.  
To reduce bias during training, we applied **SMOTE** on the training data to balance both classes.

---

## Models Trained (Supervised Learning)

We trained and compared multiple models to understand performance tradeoffs:

- **Logistic Regression** (baseline, interpretable)
- **Random Forest** (nonlinear patterns + feature importance)
- **K-Nearest Neighbors**
- **Support Vector Machine** (trained on a subset due to compute cost)
- **Neural Network (MLPClassifier)**

### Evaluation Metrics
We used metrics beyond accuracy to handle imbalance and business relevance:
- Accuracy
- Precision / Recall / F1 (especially for the positive “subscriber” class)
- ROC–AUC
- Average Precision (PR-AUC)
- Confusion Matrix

---

## Best Model

**Random Forest** was selected as the best overall model because it provided the strongest balance across metrics (not just accuracy), and also produced feature importance for interpretability.

Top important predictors included:
- `duration`
- `euribor3m`
- `nr.employed`
- `pdays`
- campaign-related features

---

## Unsupervised Learning (Segmentation)

To explore customer segments:
- **PCA** was used for dimensionality reduction
- **K-Means clustering** was applied to group customers into behavioral segments

This supports marketing strategy by showing that customers are not a single uniform group and may respond better to different approaches.

---

## Results Summary (High-level)

- Subscription likelihood varies by job category and age group
- Certain months show stronger response rates (seasonality effect)
- Longer call duration strongly correlates with subscription outcome
- Past campaign success increases future subscription likelihood
- Economic indicators contribute meaningfully to prediction performance
- Customer segmentation reveals distinct behavioral patterns

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn
- Google Colab (development environment)

---

