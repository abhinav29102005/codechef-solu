from itertools import combinations

# --------------------------------------------------
# Dataset (use items from the problem description)
# --------------------------------------------------
# Frequent 2-itemsets like:
# {"apple", "banana"}, {"apple", "milk"}
# --------------------------------------------------
L2 = {
    frozenset({"apple", "banana"}),
    frozenset({"apple", "milk"})
}

# Candidate 3-itemsets generated earlier
C3 = {
    frozenset({"apple", "banana", "milk"})
}


# Apriori Pruning Function
def prune_candidates(candidates, prev_frequent_itemsets):
    """
    Removes candidates whose any (k-1)-subset
    is NOT present in prev_frequent_itemsets
    """
    pruned = set()

    for candidate in candidates:
        k = len(candidate)

        # Generate all subsets of size (k-1)
        subsets = [frozenset(s) for s in combinations(candidate, k - 1)]

        # Keep candidate ONLY if all subsets are frequent
        if all(s in prev_frequent_itemsets for s in subsets):
            pruned.add(candidate)

    return pruned


# Apply pruning
pruned_C3 = prune_candidates(C3, L2)

print("Candidates before pruning:", C3)
print("Candidates after pruning:", pruned_C3)
