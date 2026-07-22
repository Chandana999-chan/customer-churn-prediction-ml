# 📊 Customer Churn Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave (churn) based on customer demographics, subscription details, and billing information.

Built using **Python**, **XGBoost**, **Scikit-learn**, and **Streamlit**.

---


## 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Losing customers directly impacts revenue.

This project predicts whether a customer is likely to churn using historical customer data. Businesses can use these predictions to identify at-risk customers and improve customer retention strategies.

---

## 🎯 Problem Statement

Given customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges

Predict whether the customer will:

- ✅ Stay
- ⚠️ Churn

---

## 🧠 Machine Learning Type

**Binary Classification**

Target Variable:

- 0 → Customer Stays
- 1 → Customer Churns

---

## 🤖 Machine Learning Model

**XGBoost Classifier**

Why XGBoost?

- High accuracy
- Handles complex relationships
- Fast training
- Prevents overfitting
- Widely used in industry

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## 📂 Project Structure

```
customer-churn-prediction-ml/
│
├── app/
│   └── app.py
│
├── data/
│   └── telecom_churn.csv
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Workflow

```
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Encoding
    │
    ▼
Train-Test Split
    │
    ▼
Train XGBoost Classifier
    │
    ▼
Evaluate Model
    │
    ▼
Save Model (.pkl)
    │
    ▼
Streamlit Web App
    │
    ▼
Customer Churn Prediction
```

---

## 📊 Features Used

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## 📈 Model Training

The dataset was:

- Cleaned
- Encoded
- Split into training and testing datasets

Model trained using:

```python
XGBClassifier()
```

---

## 📦 Model Saving

The trained model is saved using Joblib.

```python
joblib.dump(model, "best_model.pkl")
```

Later loaded inside the Streamlit application.

```python
model = joblib.load("best_model.pkl")
```

---

## 💻 Streamlit Application

The web application allows users to:

- Enter customer information
- Predict churn instantly
- View churn probability
- Display user-friendly inputs

---

## 📸 Application Preview

Add screenshots here after running the project.

Example:

```
screenshots/home.png

screenshots/result.png
```

---

## 📊 Future Improvements

- Use Scikit-learn Pipeline
- Hyperparameter tuning
- Cross Validation
- SHAP Explainability
- Docker Support
- Cloud Deployment
- CI/CD using GitHub Actions
- REST API using FastAPI
- Real-time Predictions

---

## 📚 Learning Outcomes

This project helped me understand:

- Data preprocessing
- Feature encoding
- Binary Classification
- XGBoost
- Model Evaluation
- Saving ML models
- Streamlit deployment
- End-to-end Machine Learning workflow

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction-ml.git
```

Move into the project

```bash
cd customer-churn-prediction-ml
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 📊 Results

The trained model successfully predicts customer churn using customer demographic and billing information through an interactive Streamlit web application.

---

## 👩‍💻 Author

**Chandana K**

Computer Science Engineering Student

Aspiring Machine Learning Engineer

---

## ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.
