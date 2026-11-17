from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Step 1: Generate dataset
def generate_data(n_samples=500, centers=4, random_state=42):
    # TODO: generate dataset using make_blobs
    X, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    return X


# Step 2: Evaluate different k values
def evaluate_kmeans(X, k_values):
    results = []  # TODO: store (k, inertia, score)

    for k in k_values:
        # TODO: initialize KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42)

        # TODO: fit model and predict labels
        labels = kmeans.fit_predict(X)

        # TODO: compute inertia
        inertia = kmeans.inertia_

        # TODO: compute silhouette score
        score = silhouette_score(X, labels)

        # TODO: append results
        results.append((k, inertia, score))

    return results


# Step 3: Train final model
def train_final_model(X, best_k):
    # TODO: initialize model
    kmeans = KMeans(n_clusters=best_k, random_state=42)

    # TODO: fit and predict
    labels = kmeans.fit_predict(X)

    return kmeans, labels


if __name__ == "__main__":
    X = generate_data()
    k_vals = [3, 4, 5, 6]
    results = evaluate_kmeans(X, k_vals)
    for k, inertia, score in results:
        print(f"k = {k}, Inertia = {inertia:.2f}, Silhouette Score = {score:.2f}")
    best_k = 4
    model, labels = train_final_model(X, best_k)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroids",
    )
    plt.title(f"K-Means Clustering with {best_k} Clusters")
    plt.legend()
    plt.savefig("clusters_eval.png")
    plt.close()
