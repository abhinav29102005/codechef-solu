import pandas as pd
import numpy as np


class HousingKMeans:
    """Implement K-Means manually using SSE convergence."""

    def __init__(
        self,
        k_min=2,
        k_max=10,
        max_iterations=100,
        tolerance=1e-4,
        n_init=5,
        random_state=42
    ):
        # Load USA_Housing.csv
        self.df = pd.read_csv("USA_Housing.csv")

        self.k_min = k_min
        self.k_max = k_max
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_init = n_init
        self.random_state = random_state

        self.X = None
        self.X_scaled = None

        self.feature_names = None
        self.feature_means = None
        self.feature_stds = None

        self.elbow_results = {}
        self.best_k = None

        self.best_centroids = None
        self.best_labels = None
        self.final_sse = None

    def prepare_data(self):
        """Use and standardize all six numerical features."""
        self.feature_names = list(self.df.columns)
        self.X = self.df.values.astype(float)

        self.feature_means = np.mean(self.X, axis=0)
        self.feature_stds = np.std(self.X, axis=0)

        # Avoid division by zero
        stds_safe = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.X_scaled = (self.X - self.feature_means) / stds_safe
        return self.X_scaled

    def initialize_centroids(self, X, k, rng):
        """Initialize K centroids using K-Means++."""
        n_samples = len(X)
        first_idx = rng.integers(0, n_samples)
        centroids = [X[first_idx]]

        for _ in range(1, k):
            # Calculate squared distance of each sample from its nearest selected centroid
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
            min_dist_sq = np.min(dists, axis=0)

            total_dist = np.sum(min_dist_sq)
            if total_dist == 0:
                probs = np.ones(n_samples) / n_samples
            else:
                probs = min_dist_sq / total_dist

            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def assign_clusters(self, X, centroids):
        """Assign samples to their nearest centroid."""
        # Calculate squared Euclidean distances: shape (n_samples, k)
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels, k, rng):
        """Recalculate centroids from cluster members."""
        new_centroids = np.zeros((k, X.shape[1]))
        n_samples = len(X)

        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = np.mean(members, axis=0)
            else:
                # Reinitialize from a random sample if cluster is empty
                rand_idx = rng.integers(0, n_samples)
                new_centroids[j] = X[rand_idx]

        return new_centroids

    def calculate_sse(self, X, labels, centroids):
        """Calculate total clustering SSE."""
        sse = np.sum((X - centroids[labels]) ** 2)
        return float(sse)

    def fit_kmeans(self, X, k):
        """Run K-Means with SSE convergence and multiple initializations."""
        best_sse = float("inf")
        best_centroids = None
        best_labels = None
        best_iters = 0
        best_history = []

        for init in range(self.n_init):
            rng = np.random.default_rng(self.random_state + init)
            centroids = self.initialize_centroids(X, k, rng)
            labels = self.assign_clusters(X, centroids)
            sse = self.calculate_sse(X, labels, centroids)
            history = [sse]

            num_iters = 0
            for it in range(self.max_iterations):
                num_iters = it + 1
                new_centroids = self.update_centroids(X, labels, k, rng)
                new_labels = self.assign_clusters(X, new_centroids)
                new_sse = self.calculate_sse(X, new_labels, new_centroids)
                history.append(new_sse)

                if abs(sse - new_sse) <= self.tolerance:
                    centroids = new_centroids
                    labels = new_labels
                    sse = new_sse
                    break

                centroids = new_centroids
                labels = new_labels
                sse = new_sse

            if sse < best_sse:
                best_sse = sse
                best_centroids = centroids
                best_labels = labels
                best_iters = num_iters
                best_history = history

        return best_centroids, best_labels, best_sse, best_iters, best_history

    def run_elbow_method(self):
        """Calculate the best SSE for each candidate K."""
        self.elbow_results = {}
        for k in range(self.k_min, self.k_max + 1):
            _, _, sse, _, _ = self.fit_kmeans(self.X_scaled, k)
            self.elbow_results[k] = sse
        return self.elbow_results

    def find_elbow(self):
        """Automatically select K from elbow results."""
        ks = sorted(self.elbow_results.keys())
        sses = [self.elbow_results[k] for k in ks]

        # Endpoints line: P1 = (k_min, sse_min), P2 = (k_max, sse_max)
        p1 = np.array([ks[0], sses[0]], dtype=float)
        p2 = np.array([ks[-1], sses[-1]], dtype=float)

        v = p2 - p1
        v_norm = np.linalg.norm(v)

        max_dist = -1.0
        best_k = ks[0]

        for i in range(len(ks)):
            p = np.array([ks[i], sses[i]], dtype=float)
            # Perpendicular distance from point p to line segment p1-p2
            # Cross product magnitude divided by line segment length
            dist = np.abs(v[0] * (p1[1] - p[1]) - (p1[0] - p[0]) * v[1]) / v_norm
            if dist > max_dist:
                max_dist = dist
                best_k = ks[i]

        self.best_k = best_k
        return self.best_k

    def fit_best_model(self):
        """Fit K-Means using the selected best K."""
        centroids, labels, sse, iters, _ = self.fit_kmeans(self.X_scaled, self.best_k)
        self.best_centroids = centroids
        self.best_labels = labels
        self.final_sse = sse

    def get_cluster_summary(self):
        """Return cluster sizes and original-scale centroids."""
        summary = {}
        for j in range(self.best_k):
            cluster_mask = (self.best_labels == j)
            num_samples = int(np.sum(cluster_mask))
            # Convert centroid back to original feature scale
            orig_centroid = self.best_centroids[j] * self.feature_stds + self.feature_means
            summary[j] = {
                "cluster_number": j,
                "number_of_samples": num_samples,
                "centroid": orig_centroid.tolist()
            }
        return summary

    def run(self):
        """Run the complete clustering workflow."""
        self.prepare_data()
        self.run_elbow_method()
        self.find_elbow()
        self.fit_best_model()
        cluster_summary = self.get_cluster_summary()

        return {
            "elbow_results": self.elbow_results,
            "best_k": self.best_k,
            "final_sse": self.final_sse,
            "cluster_summary": cluster_summary
        }

    def get_dataset(self):
        """Return the originally loaded DataFrame."""
        return self.df


if __name__ == "__main__":
    model = HousingKMeans()
    results = model.run()

    print("Elbow Results:")
    print(results["elbow_results"])

    print("\nBest K:")
    print(results["best_k"])

    print("\nFinal SSE:")
    print(results["final_sse"])

    print("\nCluster Summary:")
    print(results["cluster_summary"])
