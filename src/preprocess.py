import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Clean and preprocess the customer churn dataset.
    """

    # Remove customerID (not useful for prediction)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()

    for col in categorical_columns:
        if col != "Churn":
            df[col] = encoder.fit_transform(df[col])

    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


if __name__ == "__main__":

    file_path = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_data(file_path)

    X, y = preprocess_data(df)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nFirst Five Rows")
    print(X.head())

    print("\nTarget Distribution")
    print(y.value_counts())