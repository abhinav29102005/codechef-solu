import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


class NaiveBayesClassifier:

    def __init__(self):
        """
        Load dataset.
        """
        print("Loading dataset...")
        self.df = pd.read_csv("dataset.csv")

    def prepare_data(self):
        """
        Prepare features and target class.
        """
        # Feature attributes (all except last column)
        # Target class (Species or last column)
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # Split into training data and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train Naive Bayes classifier.
        """
        print("\nTraining Naive Bayes classifier...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        """
        Generate predictions.
        """
        return model.predict(X_test)

    def evaluate(self, y_test, predictions):
        """
        Evaluate model performance.
        """
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        print("Model Accuracy:")
        print(acc)
        print("\nClassification Report:")
        print(report)


def run_naive_bayes():
    model = NaiveBayesClassifier()
    X_train, X_test, y_train, y_test = model.prepare_data()
    classifier = model.train_model(X_train, y_train)
    predictions = model.predict(classifier, X_test)
    model.evaluate(y_test, predictions)


if __name__ == "__main__":
    run_naive_bayes()
