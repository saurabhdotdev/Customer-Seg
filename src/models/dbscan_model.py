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
        """
        best_silhouette = -1.0
        best_params = {'eps': self.eps, 'min_samples': self.min_samples}
        grid_results = []

        for eps in eps_values:
            for ms in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=ms)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                if n_clusters > 1 and noise_ratio < 0.35:  # ensure reasonable cluster structure
                    # Evaluate metrics on non-noise points
                    valid_mask = labels != -1
                    if len(set(labels[valid_mask])) > 1:
                        sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
                        db = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
                        ch = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
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
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        unique_labels = set(self.labels_) - {-1}
        if not unique_labels:
            return np.array([-1] * len(X_sample))
            
        centroids = {l: X_train[self.labels_ == l].mean(axis=0) for l in unique_labels}
        
        predictions = []
        for x in X_sample:
            dists = {l: np.linalg.norm(x - centroids[l]) for l in centroids}
            min_cluster = min(dists, key=dists.get)
            min_dist = dists[min_cluster]
            
            # If distance exceeds 2*eps, classify as noise
            if min_dist > 2.5 * self.eps:
                predictions.append(-1)
            else:
                predictions.append(min_cluster)
                
        return np.array(predictions)
