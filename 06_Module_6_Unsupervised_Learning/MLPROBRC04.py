import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class FlowerSpeciesClassification:
    """
    Build a Bayesian Classification model
    for flower species prediction.
    """

    def __init__(self):
        """
        Load the dataset from
        flower_species_classification.csv.
        """
        self.df = pd.read_csv("flower_species_classification.csv")

    def clean_dataset(self):
        """
        Remove:
        - Duplicate rows
        - Rows containing missing values

        Store the cleaned dataset.
        """
        self.df = self.df.drop_duplicates().dropna()

    def perform_eda(self):
        """
        Generate a summary containing:
        - Total records
        - Number of features
        - Missing values
        - Duplicate rows
        """
        summary = {
            "Total records": len(self.df),
            "Number of features": self.df.shape[1],
            "Missing values": int(self.df.isnull().sum().sum()),
            "Duplicate rows": int(self.df.duplicated().sum()),
        }
        return summary

    def transform_dataset(self):
        """
        Separate:
        - Features (X)
        - Target (y)
        Store them for later use.
        """
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

    def train_model(self):
        """
        Split the cleaned dataset into training
        and testing sets.

        Train a Gaussian Naive Bayes classifier.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

    def evaluate_model(self):
        """
        Calculate:
        - Accuracy
        - Classification Report
        """
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        report = classification_report(self.y_test, preds)
        return {"Accuracy": acc, "Classification Report": report}

    def predict_species(
        self, sepal_length, sepal_width, petal_length, petal_width
    ):
        """
        Predict the flower species.
        """
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model.predict(sample)
        return prediction[0]

    def get_dataset(self):
        """
        Return the cleaned dataset.
        """
        return self.df


if __name__ == "__main__":
    model = FlowerSpeciesClassification()

    model.clean_dataset()
    model.perform_eda()
    model.transform_dataset()
    model.train_model()

    print("EDA Summary:")
    print(model.perform_eda())

    print("\nEvaluation Metrics:")
    print(model.evaluate_model())

    print("\nPrediction:")
    print(model.predict_species(6.2, 2.9, 4.3, 1.3))
