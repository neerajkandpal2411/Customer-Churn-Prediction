# 📊 Customer Churn Prediction using Machine Learning

This project predicts customer churn in a telecommunications company using supervised machine learning models. We use two powerful algorithms — **Random Forest** and **XGBoost** — and compare their performance based on evaluation metrics like **Accuracy** and **ROC AUC**.

---

## 🧠 Problem Statement

Customer churn is a critical problem in the telecom industry. The goal is to identify which customers are likely to leave the service, allowing the company to take necessary actions to retain them.

---

## 📌 Dataset

- **Source**: [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contains information about telecom users: demographics, services used, account details, and churn label.

---

## 🚀 Workflow Overview

1. **Data Cleaning & EDA** – Handled missing values and visualized key patterns.
2. **Feature Engineering** – Encoded categorical variables and scaled numerical ones.
3. **Model Building** – Built and tuned Random Forest and XGBoost models.
4. **Evaluation** – Compared models using Accuracy and ROC AUC on test data.

---

## 📈 Results

| Model         | Accuracy | ROC AUC |
|---------------|----------|---------|
| Random Forest | 0.7939   | 0.8319  |
| XGBoost       | 0.7967   | 0.8354  |

✅ **Conclusion**: XGBoost performed slightly better based on ROC AUC and was chosen as the final model.

---

## 💻 How to Run

```bash
 git clone https://github.com/vedaansh12/customer_churn_prediction.git
 cd customer_churn_prediction

 python -m venv venv
 venv\Scripts\activate    # Windows
 source venv/bin/activate # macOS/Linux

 pip install -r requirements.txt
 streamlit run app.py
```

---



**🙋‍♂️ Author**: <br>
Vedaansh Vishwakarma <br>
LinkedIn : (www.linkedin.com/in/vedaansh-vishwakarma-057a7124b)
=======
