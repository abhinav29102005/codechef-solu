from itertools import combinations
from collections import Counter

# --------------------------------------------------
# Dataset
# --------------------------------------------------
# Transactions like:
# {"apple", "banana", "milk"}, etc.
# --------------------------------------------------
transactions = [
    {"apple", "banana", "milk"},
    {"apple", "banana"},
    {"banana", "milk"},
    {"apple", "milk"},
    {"apple", "banana", "milk"}
]

min_support = 3


# Step 1: Find frequent 1-itemsets
def find_frequent_1_itemsets(transactions, min_support):
    # Hint:
    # - Count how many times each item appears
    # - Keep items meeting min_support
    item_counts = Counter(item for t in transactions for item in t)

    return {
        frozenset([item])
        for item, count in item_counts.items()
        if count >= min_support
    }


# Step 2: Generate candidate (k+1)-itemsets
def generate_candidates(prev_frequent_itemsets, k):
    # Hint:
    # - Combine frequent k-itemsets
    # - Generate itemsets of size (k+1)
    items = set().union(*prev_frequent_itemsets)

    return {
        frozenset(combo)
        for combo in combinations(items, k + 1)
    }


# Step 3: Count support of an itemset
def support_count(itemset, transactions):
    # Hint:
    # - Count how many transactions contain this itemset
    return sum(1 for t in transactions if itemset.issubset(t))


# Step 4: Filter candidates by support
def filter_frequent_itemsets(candidates, transactions, min_support):
    return {
        itemset
        for itemset in candidates
        if support_count(itemset, transactions) >= min_support
    }


# Run Apriori steps
L1 = find_frequent_1_itemsets(transactions, min_support)
C2 = generate_candidates(L1, k=1)
L2 = filter_frequent_itemsets(C2, transactions, min_support)
C3 = generate_candidates(L2, k=2)

print("Frequent 1-itemsets:", L1)
print("Frequent 2-itemsets:", L2)
print("Candidate 3-itemsets:", C3)
