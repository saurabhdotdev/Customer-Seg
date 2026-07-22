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
        Fits Agglomerative Clustering on dataset X with sub-sampling optimization for large N.
        """
        if len(X) > 10000:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), size=10000, replace=False)
            X_sub = X[indices]
            sub_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
            sub_labels = sub_model.fit_predict(X_sub)
            self.labels_ = self.predict_nearest_cluster(X, X_sub, sub_labels)
            self.model = sub_model
        else:
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
            self.labels_ = self.model.fit_predict(X)
        return self

    def predict_nearest_cluster(self, X_sample: np.ndarray, X_train: np.ndarray, train_labels: np.ndarray = None) -> np.ndarray:
        """
        Assigns new points to the nearest cluster centroid of fitted train matrix.
        """
        labels = self.labels_ if train_labels is None else train_labels
        unique_labels = set(labels)
        centroids = {l: X_train[labels == l].mean(axis=0) for l in unique_labels}
        
        predictions = []
        for x in X_sample:
            dists = {l: np.linalg.norm(x - centroids[l]) for l in centroids}
            predictions.append(min(dists, key=dists.get))
            
        return np.array(predictions)
