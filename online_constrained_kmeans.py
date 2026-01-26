import numpy as np

class OnlineConstrainedKMeans:
    """
    Clustering online (procesa 1 muestra a la vez) con restricción de tamaño máximo por cluster.
    - Asigna al centroide más cercano con cupo disponible.
    - Actualiza centroide incrementalmente.
    """
    def __init__(self, n_clusters, max_sizes, random_state=42):
        self.k = int(n_clusters)
        self.max_sizes = np.array(max_sizes, dtype=int)
        assert len(self.max_sizes) == self.k
        self.random_state = int(random_state)

        self.centroids_ = None
        self.counts_ = np.zeros(self.k, dtype=int)

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids_ = X[idx].astype(np.float32)

    def partial_fit_predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.centroids_ is None:
            self._init_centroids(X)

        labels = -np.ones(len(X), dtype=int)

        for i, x in enumerate(X):
            dists = np.linalg.norm(self.centroids_ - x, axis=1)
            order = np.argsort(dists)

            chosen = None
            for c in order:
                if self.counts_[c] < self.max_sizes[c]:
                    chosen = c
                    break

            if chosen is None:
                chosen = order[0]  # fallback

            labels[i] = chosen

            self.counts_[chosen] += 1
            eta = 1.0 / self.counts_[chosen]
            self.centroids_[chosen] = (1 - eta) * self.centroids_[chosen] + eta * x

        return labels
