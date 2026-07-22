import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class KMeansSegmentationModel:
    """
    K-Means Clustering implementation with optimal K grid search.
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels_ = None
        self.cluster_centers_ = None

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        """
        Evaluates K values from k_range and returns inertia and silhouette scores.
        """
        results = []
        best_k = self.n_clusters
        best_silhouette = -1.0
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertia = km.inertia_
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            results.append({
                'k': k,
                'inertia': float(inertia),
                'silhouette_score': float(sil),
                'davies_bouldin': float(db),
                'calinski_harabasz': float(ch)
            })
            
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

        return {
            'best_k': best_k,
            'best_silhouette': best_silhouette,
            'grid_search': results
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
