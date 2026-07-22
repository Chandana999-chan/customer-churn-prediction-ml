import joblib
import pandas as pd

model = joblib.load("../models/best_model.pkl")

sample = {
    "gender":1,
    "SeniorCitizen":0,
    "Partner":1,
    "Dependents":0,
    "tenure":20,
    "PhoneService":1,
    "MultipleLines":0,
    "InternetService":1,
    "OnlineSecurity":0,
    "OnlineBackup":1,
    "DeviceProtection":1,
    "TechSupport":0,
    "StreamingTV":1,
    "StreamingMovies":1,
    "Contract":0,
    "PaperlessBilling":1,
    "PaymentMethod":2,
    "MonthlyCharges":80,
    "TotalCharges":1800
}

df = pd.DataFrame([sample])

prediction = model.predict(df)

if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Stay")