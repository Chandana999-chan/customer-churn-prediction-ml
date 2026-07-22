import os
import joblib

from preprocess import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ===============================
# Load Dataset
# ===============================

file_path = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = load_data(file_path)

X, y = preprocess_data(df)

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Models
# ===============================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
}

best_model = None
best_accuracy = 0

print("=" * 60)
print("Training Models...")
print("=" * 60)

for name, model in models.items():

    print(f"\n{name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# ===============================
# Save Best Model
# ===============================

os.makedirs("../models", exist_ok=True)

joblib.dump(best_model, "../models/best_model.pkl")

print("\n" + "=" * 60)
print(f"Best Accuracy : {best_accuracy:.4f}")
print("Best model saved successfully!")
print("=" * 60)