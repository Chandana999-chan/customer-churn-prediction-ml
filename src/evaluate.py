import joblib
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load data
df = load_data("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("../models/best_model.pkl")

predictions = model.predict(X_test)

# Create images folder
import os
os.makedirs("../images", exist_ok=True)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.savefig("../images/confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.savefig("../images/roc_curve.png")
plt.close()

# Feature Importance
if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    plt.figure(figsize=(10,6))
    plt.barh(X.columns, importance)
    plt.tight_layout()
    plt.savefig("../images/feature_importance.png")
    plt.close()

print("Evaluation Complete")