import pandas as pd


class NaiveBayesClassifier:

    def __init__(self, filepath):
        """
        Load dataset from CSV.
        """
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)

    def compute_prior(self):
        """
        Compute prior probabilities of each class.
        """
        target_col = self.df.columns[-1]
        total = len(self.df)
        priors = (self.df[target_col].value_counts() / total).to_dict()
        return priors

    def conditional_probability(self, column, value, class_label):
        """
        Compute P(feature=value | class).
        """
        target_col = self.df.columns[-1]
        subset = self.df[self.df[target_col] == class_label]
        if len(subset) == 0:
            return 0.0
        count = (subset[column] == value).sum()
        return count / len(subset)

    def predict(self, sample):
        """
        Predict class label using Naive Bayes.
        """
        priors = self.compute_prior()
        probs = {}
        for class_label, prior in priors.items():
            prob = prior
            for col, val in sample.items():
                if col in self.df.columns:
                    prob *= self.conditional_probability(col, val, class_label)
            probs[class_label] = prob
        return max(probs, key=probs.get)


def run_classifier():
    model = NaiveBayesClassifier("allelectronics.csv")

    sample = {
        "age": "youth",
        "income": "medium",
        "student": "yes",
        "credit_rating": "fair",
    }

    prediction = model.predict(sample)
    return prediction


if __name__ == "__main__":
    result = run_classifier()
    print("\nPredicted Class:", result)
