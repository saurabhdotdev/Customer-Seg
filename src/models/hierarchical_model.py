import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class HierarchicalSegmentationModel:
    """
    Agglomerative Hierarchical Clustering implementation.
    """
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = None

    def fit(self, X: np.ndarray):
        """
        Fits Agglomerative Clustering on dataset X.
        """
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = self.model.fit_predict(X)
        return self

    def predict_nearest_cluster(self, X_sample: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Assigns new points to the nearest cluster centroid of fitted train matrix.
        """
        unique_labels = set(self.labels_)
        centroids = {l: X_train[self.labels_ == l].mean(axis=0) for l in unique_labels}
        
        predictions = []
        for x in X_sample:
            dists = {l: np.linalg.norm(x - centroids[l]) for l in centroids}
            predictions.append(min(dists, key=dists.get))
            
        return np.array(predictions)
