import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

class KMeansSegmentationModel:
    """
    K-Means Clustering implementation with optimal K grid search and silhouette sample analysis.
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels_ = None
        self.cluster_centers_ = None

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        """
        Evaluates K values from k_range and returns inertia, silhouette scores, DB index, and CH index.
        """
        results = []
        best_k = self.n_clusters
        best_silhouette = -1.0
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(X)
            inertia = float(km.inertia_)
            sample_sz = 10000 if len(X) > 10000 else None
            sil = float(silhouette_score(X, labels, sample_size=sample_sz, random_state=42))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            
            results.append({
                'k': int(k),
                'inertia': round(inertia, 2),
                'silhouette_score': round(sil, 4),
                'davies_bouldin': round(db, 4),
                'calinski_harabasz': round(ch, 2)
            })
            
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

        return {
            'best_k': int(best_k),
            'best_silhouette': round(best_silhouette, 4),
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
        Fits K-Means model on dataset X.
        """
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts cluster assignment for input matrix X.
        """
        return self.model.predict(X)
