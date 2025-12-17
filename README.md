# 🏦 Bank Churn Prediction

## 🚀 Overview
This project is a machine learning solution designed to predict which bank customers are likely to leave (churn). By identifying "at-risk" customers early, banks can take proactive steps (like offering better rates) to retain them, directly impacting the bottom line.

## 🎯 Problem Statement
Customer acquisition is **5x more expensive** than customer retention. The goal of this project is to classify customers into two categories:
1.  **Exited (1):** Customer left the bank.
2.  **Retained (0):** Customer stayed.

**Challenge:** The data was highly imbalanced (mostly retained customers), requiring specific techniques to avoid bias.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
* **Model:** XGBoost Classifier & Random Forest
* **Technique:** SMOTE (Synthetic Minority Over-sampling Technique) for balancing data.
* <img width="1012" height="541" alt="image" src="https://github.com/user-attachments/assets/8e01a6a2-8a6f-4d16-97f6-02acd8374ac9" />


## ⚙️ Workflow
1.  **Data Preprocessing:**
    * Handled missing values.
    * Encoded categorical variables (e.g., "Gender", "Geography") using One-Hot Encoding.
2.  **Feature Engineering:**
    * Analyzed correlations between `Age`, `CreditScore`, and `Balance`.
    * Identified that older customers with higher balances were more likely to churn.
3.  **Handling Imbalance:**
    * Used **SMOTE** to generate synthetic samples for the "Exited" class so the model wouldn't just guess "Retained" every time.
4.  **Model Training:**
    * Compared Logistic Regression vs. XGBoost.
    * Selected **XGBoost** for its superior performance on non-linear data.

## 📊 Key Results
* **Accuracy:** 86%
* **Recall (The most important metric):** 80%
    * *Why Recall?* In banking, it is worse to miss a customer who is about to leave (False Negative) than to accidentally flag a loyal customer (False Positive).

## 🚀 How to Run
1.  **Install requirements**
    ```bash
    pip install pandas sklearn xgboost seaborn
    ```
2.  **Run the script**
    ```bash
    python churn_prediction.py
    ```

---
*Project developed for Financial Data Analysis study.*
