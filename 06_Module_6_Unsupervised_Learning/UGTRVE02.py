import pandas as pd
import numpy as np


class HousingKMedoids:
    """Implement K-Medoids manually using SSE convergence."""

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

        self.best_medoid_indices = None
        self.best_medoids = None
        self.best_labels = None
        self.final_sse = None

    def prepare_data(self):
        """Prepare and standardize all six numerical features."""
        self.feature_names = list(self.df.columns)
        self.X = self.df.values.astype(float)

        self.feature_means = np.mean(self.X, axis=0)
        self.feature_stds = np.std(self.X, axis=0)

        # Avoid zero division
        stds_safe = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.X_scaled = (self.X - self.feature_means) / stds_safe
        return self.X_scaled

    def initialize_medoids(self, X, k, rng):
        """Select K unique initial medoid indices."""
        n_samples = len(X)
        medoid_indices = rng.choice(n_samples, size=k, replace=False)
        return list(medoid_indices)

    def assign_clusters(self, X, medoid_indices):
        """Assign samples to nearest medoids."""
        medoids = X[medoid_indices]
        # Calculate squared Euclidean distances: shape (n_samples, k)
        distances = np.sum((X[:, np.newaxis, :] - medoids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def calculate_sse(self, X, labels, medoid_indices):
        """Calculate the total Sum of Squared Errors."""
        medoids = X[medoid_indices]
        sse = np.sum((X - medoids[labels]) ** 2)
        return float(sse)

    def find_cluster_medoid(self, X, cluster_indices):
        """
        Given indices of all samples belonging to one cluster,
        find the sample that minimizes total squared distance to all other members.
        """
        if len(cluster_indices) == 1:
            return cluster_indices[0]

        cluster_points = X[cluster_indices]
        # Pairwise squared distances: shape (n_cluster, n_cluster)
        diff = cluster_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :]
        pairwise_dist = np.sum(diff ** 2, axis=2)
        total_distances = np.sum(pairwise_dist, axis=1)

        best_idx_in_cluster = np.argmin(total_distances)
        return cluster_indices[best_idx_in_cluster]

    def update_medoids(self, X, labels, k, current_medoids, rng):
        """
        Update medoid for each cluster. If empty, select an unused random sample.
        """
        new_medoid_indices = []
        n_samples = len(X)

        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) > 0:
                best_medoid_idx = self.find_cluster_medoid(X, members)
                new_medoid_indices.append(best_medoid_idx)
            else:
                # Pick a random unused sample index
                available = list(set(range(n_samples)) - set(new_medoid_indices))
                rand_idx = rng.choice(available)
                new_medoid_indices.append(rand_idx)

        return new_medoid_indices

    def fit_kmedoids(self, X, k):
        """Run K-Medoids for a supplied value of K with multiple initializations."""
        best_sse = float("inf")
        best_medoid_indices = None
        best_labels = None
        best_iters = 0
        best_history = []

        for init in range(self.n_init):
            rng = np.random.default_rng(self.random_state + init)
            medoid_indices = self.initialize_medoids(X, k, rng)
            labels = self.assign_clusters(X, medoid_indices)
            sse = self.calculate_sse(X, labels, medoid_indices)
            history = [sse]

            num_iters = 0
            for it in range(self.max_iterations):
                num_iters = it + 1
                new_medoid_indices = self.update_medoids(X, labels, k, medoid_indices, rng)
                new_labels = self.assign_clusters(X, new_medoid_indices)
                new_sse = self.calculate_sse(X, new_labels, new_medoid_indices)
                history.append(new_sse)

                # Check convergence: tolerance or medoid indices unchanged
                if abs(sse - new_sse) <= self.tolerance or sorted(medoid_indices) == sorted(new_medoid_indices):
                    medoid_indices = new_medoid_indices
                    labels = new_labels
                    sse = new_sse
                    break

                medoid_indices = new_medoid_indices
                labels = new_labels
                sse = new_sse

            if sse < best_sse:
                best_sse = sse
                best_medoid_indices = medoid_indices
                best_labels = labels
                best_iters = num_iters
                best_history = history

        medoid_values = X[best_medoid_indices]
        return best_medoid_indices, medoid_values, best_labels, best_sse, best_iters, best_history

    def run_elbow_method(self):
        """Calculate the lowest final SSE for each candidate K."""
        self.elbow_results = {}
        for k in range(self.k_min, self.k_max + 1):
            _, _, _, sse, _, _ = self.fit_kmedoids(self.X_scaled, k)
            self.elbow_results[k] = sse
        return self.elbow_results

    def find_elbow(self):
        """Automatically select K using elbow geometry (perpendicular distance)."""
        ks = sorted(self.elbow_results.keys())
        sses = [self.elbow_results[k] for k in ks]

        p1 = np.array([ks[0], sses[0]], dtype=float)
        p2 = np.array([ks[-1], sses[-1]], dtype=float)

        v = p2 - p1
        v_norm = np.linalg.norm(v)

        max_dist = -1.0
        best_k = ks[0]

        for i in range(len(ks)):
            p = np.array([ks[i], sses[i]], dtype=float)
            dist = np.abs(v[0] * (p1[1] - p[1]) - (p1[0] - p[0]) * v[1]) / v_norm
            if dist > max_dist:
                max_dist = dist
                best_k = ks[i]

        self.best_k = best_k
        return self.best_k

    def fit_best_model(self):
        """Fit K-Medoids using best K."""
        medoid_indices, medoids, labels, sse, iters, _ = self.fit_kmedoids(self.X_scaled, self.best_k)
        self.best_medoid_indices = medoid_indices
        self.best_medoids = medoids
        self.best_labels = labels
        self.final_sse = sse

    def get_cluster_summary(self):
        """Return final cluster information with actual dataset rows."""
        summary = {}
        for j in range(self.best_k):
            cluster_mask = (self.best_labels == j)
            num_samples = int(np.sum(cluster_mask))
            medoid_idx = int(self.best_medoid_indices[j])
            # Return medoid values on original scale (from self.X)
            orig_medoid_values = self.X[medoid_idx]

            summary[j] = {
                "cluster_number": j,
                "number_of_samples": num_samples,
                "medoid_row_index": medoid_idx,
                "medoid_values": orig_medoid_values.tolist()
            }
        return summary

    def run(self):
        """Run the complete K-Medoids workflow."""
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
        """Return originally loaded DataFrame."""
        return self.df


if __name__ == "__main__":
    model = HousingKMedoids()
    results = model.run()

    print("Elbow Results:")
    print(results["elbow_results"])

    print("\nBest K:")
    print(results["best_k"])

    print("\nFinal SSE:")
    print(results["final_sse"])

    print("\nCluster Summary:")
    print(results["cluster_summary"])
