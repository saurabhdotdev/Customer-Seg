import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class DBSCANSegmentationModel:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    implementation with hyperparameter grid search for epsilon and min_samples.
    """
    def __init__(self, eps: float = 1.5, min_samples: int = 15):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = None

    def tune_hyperparameters(self, X: np.ndarray, eps_values: list = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0], min_samples_values: list = [10, 15, 20, 25]) -> dict:
        """
        Performs grid search over eps and min_samples parameters.
        Optimized with sub-sampling for large N.
        """
        best_silhouette = -1.0
        best_params = {'eps': self.eps, 'min_samples': self.min_samples}
        grid_results = []

        X_tune = X
        if len(X) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), size=10000, replace=False)
            X_tune = X[idx]

        for eps in eps_values:
            for ms in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=ms)
                labels = dbscan.fit_predict(X_tune)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                if n_clusters > 1 and noise_ratio < 0.35:  # ensure reasonable cluster structure
                    # Evaluate metrics on non-noise points
                    valid_mask = labels != -1
                    if len(set(labels[valid_mask])) > 1:
                        sample_sz = 10000 if len(X_tune[valid_mask]) > 10000 else None
                        sil = float(silhouette_score(X_tune[valid_mask], labels[valid_mask], sample_size=sample_sz, random_state=42))
                        db = float(davies_bouldin_score(X_tune[valid_mask], labels[valid_mask]))
                        ch = float(calinski_harabasz_score(X_tune[valid_mask], labels[valid_mask]))
                    else:
                        sil, db, ch = -1.0, 999.0, 0.0
                else:
                    sil, db, ch = -1.0, 999.0, 0.0

                grid_results.append({
                    'eps': eps,
                    'min_samples': ms,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'silhouette_score': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch
                })

                if sil > best_silhouette:
                    best_silhouette = sil
                    best_params = {'eps': eps, 'min_samples': ms}

        return {
            'best_params': best_params,
            'best_silhouette': best_silhouette,
            'grid_search': grid_results
        }

    def fit(self, X: np.ndarray):
        """
        Fits DBSCAN model on dataset X.
        """
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X)
        return self

    def predict_nearest_cluster(self, X_sample: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't natively have a .predict() for new points.
        This assigns new points to the nearest non-noise cluster centroid or noise (-1).
        Vectorized with NumPy for ultra-fast performance.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        unique_labels = sorted(list(set(self.labels_) - {-1}))
        if not unique_labels:
            return np.array([-1] * len(X_sample))
            
        centroids = np.array([X_train[self.labels_ == l].mean(axis=0) for l in unique_labels])
        dists = np.linalg.norm(X_sample[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        min_indices = np.argmin(dists, axis=1)
        min_distances = np.min(dists, axis=1)

        predictions = np.array([unique_labels[idx] if min_distances[i] <= 2.5 * self.eps else -1 for i, idx in enumerate(min_indices)])
        return predictions

