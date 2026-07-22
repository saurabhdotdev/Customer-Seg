import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

class DimensionalityReducer:
    """
    Computes PCA and t-SNE projections in 2D and 3D space for web visualization.
    """
    def __init__(self, n_components_pca: int = 3, pca_save_path: str = None):
        self.n_components_pca = n_components_pca
        self.pca_save_path = pca_save_path
        self.pca = PCA(n_components=n_components_pca, random_state=42)

    def fit_transform_pca(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Fits PCA and transforms dataset X. Returns 3D matrix and variance ratio metadata.
        """
        X_pca = self.pca.fit_transform(X)
        var_ratios = self.pca.explained_variance_ratio_.tolist()
        total_var = sum(var_ratios)
        
        metadata = {
            'variance_explained_per_component': [round(v, 4) for v in var_ratios],
            'total_variance_explained': round(total_var, 4)
        }
        
        if self.pca_save_path:
            os.makedirs(os.path.dirname(self.pca_save_path), exist_ok=True)
            joblib.dump(self.pca, self.pca_save_path)
            print(f"Saved PCA model to {self.pca_save_path}")

        return X_pca, metadata

    def transform_single_customer_pca(self, X_single: np.ndarray, pca_path: str = None) -> list:
        """
        Transforms a single scaled vector into 3D PCA space.
        """
        if pca_path and os.path.exists(pca_path):
            pca_model = joblib.load(pca_path)
            coord = pca_model.transform(X_single)[0]
        else:
            coord = self.pca.transform(X_single)[0]
            
        return [round(float(c), 4) for c in coord]

    @staticmethod
    def compute_tsne_2d(X: np.ndarray, sample_size: int = 1500) -> np.ndarray:
        """
        Computes 2D t-SNE projection on sample (for web rendering efficiency).
        """
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sub = X[indices]
        else:
            X_sub = X
            
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        return tsne.fit_transform(X_sub)
