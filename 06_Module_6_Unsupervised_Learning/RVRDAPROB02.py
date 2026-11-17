import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


class AssociationRuleMiningAnalysis:
    """
    Perform Association Rule Mining
    using Apriori and FP-Growth.
    """

    def __init__(self):
        """
        Load the dataset from
        market_basket_transactions.csv.
        """
        self.dataset = pd.read_csv("market_basket_transactions.csv")
        self.prepare_transactions()

    def prepare_transactions(self):
        """
        Convert the dataset into
        transaction baskets.

        Return the transaction list.
        """
        # Group items by TransactionID
        transactions = (
            self.dataset.groupby("TransactionID")["Item"].apply(list).tolist()
        )
        self.transactions = transactions

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

        return transactions

    def generate_apriori_itemsets(self, min_support):
        """
        Generate frequent itemsets
        using the Apriori algorithm.

        Store the generated itemsets
        for later retrieval.
        """
        self.apriori_itemsets = apriori(
            self.encoded_df, min_support=min_support, use_colnames=True
        )
        return self.apriori_itemsets

    def generate_fpgrowth_itemsets(self, min_support):
        """
        Generate frequent itemsets
        using the FP-Growth algorithm.

        Store the generated itemsets
        for later retrieval.
        """
        self.fpgrowth_itemsets = fpgrowth(
            self.encoded_df, min_support=min_support, use_colnames=True
        )
        return self.fpgrowth_itemsets

    def generate_association_rules(self, min_confidence):
        """
        Generate association rules
        containing:

        - Antecedent
        - Consequent
        - Support
        - Confidence
        - Lift
        """
        if not hasattr(self, "apriori_itemsets") or self.apriori_itemsets.empty:
            self.generate_apriori_itemsets(min_support=0.25)

        rules = association_rules(
            self.apriori_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        return rules

    def compare_algorithm_performance(self, min_support):
        """
        Compare:

        - Number of itemsets generated
          by Apriori

        - Number of itemsets generated
          by FP-Growth
        """
        ap_sets = self.generate_apriori_itemsets(min_support)
        fp_sets = self.generate_fpgrowth_itemsets(min_support)
        return {
            "Apriori Itemsets": len(ap_sets),
            "FP-Growth Itemsets": len(fp_sets),
        }

    def get_dataset(self):
        """
        Return the loaded dataset.
        """
        return self.dataset

    def generate_analysis_summary(self):
        """
        Generate a summary containing:

        - Total transactions
        - Total unique items
        - Total Apriori itemsets
        - Total FP-Growth itemsets
        """
        total_tx = (
            len(self.transactions)
            if hasattr(self, "transactions")
            else self.dataset["TransactionID"].nunique()
        )
        total_items = self.dataset["Item"].nunique()
        total_ap = (
            len(self.apriori_itemsets)
            if hasattr(self, "apriori_itemsets")
            else 0
        )
        total_fp = (
            len(self.fpgrowth_itemsets)
            if hasattr(self, "fpgrowth_itemsets")
            else 0
        )

        return {
            "Total transactions": total_tx,
            "Total unique items": total_items,
            "Total Apriori itemsets": total_ap,
            "Total FP-Growth itemsets": total_fp,
        }


if __name__ == "__main__":
    analysis = AssociationRuleMiningAnalysis()

    analysis.generate_apriori_itemsets(min_support=0.25)
    analysis.generate_fpgrowth_itemsets(min_support=0.25)

    print("Analysis Summary:")
    print(analysis.generate_analysis_summary())

    print("\nAssociation Rules:")
    print(analysis.generate_association_rules(min_confidence=0.5))
