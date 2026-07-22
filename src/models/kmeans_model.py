import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

class KMeansSegmentationModel:
    """
    K-Means & MiniBatchKMeans Clustering implementation with optimal K grid search and silhouette sample analysis.
    Auto-switches to MiniBatchKMeans for large datasets (n_samples > 50,000) for ultra-fast performance.
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_minibatch = False

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        """
        Evaluates K values from k_range and returns inertia, silhouette scores, DB index, and CH index.
        """
        results = []
        best_k = self.n_clusters
        best_silhouette = -1.0
        
        max_k_allowed = min(len(X) - 1, max(k_range))
        valid_k_range = [k for k in k_range if 2 <= k <= max_k_allowed]
        if not valid_k_range:
            valid_k_range = [2] if len(X) >= 2 else [1]
        
        use_minibatch = len(X) > 50000

        for k in valid_k_range:
            if use_minibatch:
                km = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, batch_size=2048, n_init=3)
            else:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)

            labels = km.fit_predict(X)
            inertia = float(km.inertia_)
            sample_sz = 10000 if len(X) > 10000 else None
            sil = float(silhouette_score(X, labels, sample_size=sample_sz, random_state=42)) if len(set(labels)) > 1 else 0.0
            db = float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else 0.0
            ch = float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else 0.0
            
            results.append({
                'k': int(k),
                'inertia': round(inertia, 2),
                'silhouette_score': round(sil, 4),
                'davies_bouldin': round(db, 4),
                'calinski_harabasz': round(ch, 2),
                'is_minibatch': use_minibatch
            })
            
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

        return {
            'best_k': int(best_k),
            'best_silhouette': round(best_silhouette, 4),
            'is_minibatch': use_minibatch,
            'grid_search': results
        }

    def compute_silhouette_samples_data(self, X: np.ndarray) -> dict:
        """
        Computes silhouette coefficients per sample for visual silhouette plots.
        """
        if self.labels_ is None:
            self.fit(X)
            
        sample_sil_values = silhouette_samples(X, self.labels_)
        avg_sil = float(silhouette_score(X, self.labels_))
        
        cluster_silhouettes = {}
        for c in range(self.n_clusters):
            c_vals = sample_sil_values[self.labels_ == c]
            c_vals.sort()
            cluster_silhouettes[int(c)] = [round(float(v), 4) for v in c_vals[::max(1, len(c_vals)//100)]] # Sample 100 points
            
        return {
            'average_silhouette': round(avg_sil, 4),
            'cluster_silhouettes': cluster_silhouettes
        }

    def fit(self, X: np.ndarray):
        """
        Fits K-Means or MiniBatchKMeans model on dataset X.
        """
        if len(X) > 50000:
            self.is_minibatch = True
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=2048,
                n_init=5
            )
        else:
            self.is_minibatch = False
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts cluster assignment for input matrix X.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calling predict.")
        return self.model.predict(X)
