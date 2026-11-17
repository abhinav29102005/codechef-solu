import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    return df


# --------------------------------------------------
# 2. Feature Selection & Target Encoding
# --------------------------------------------------
def select_features(df):
    target_col = "variety" if "variety" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    return X, y


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------
def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# 5. Predict
# --------------------------------------------------
def predict(model, X_test):
    return model.predict(X_test)


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------
def evaluate_model(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    DATASET_PATH = "dataset.csv"

    # Load dataset
    df = load_dataset(DATASET_PATH)

    print(df.head())

    # Feature selection and target encoding
    X, y = select_features(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Predict
    y_pred = predict(model, X_test)

    # Accuracy
    accuracy = evaluate_model(y_test, y_pred)
    print("\nAccuracy of Naive Bayes classifier:", round(accuracy * 100, 2), "%")
