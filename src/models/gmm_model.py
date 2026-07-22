import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class GMMSegmentationModel:
    """
    Gaussian Mixture Model (GMM) soft-clustering implementation with BIC/AIC search.
    """
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels_ = None
        self.means_ = None

    def find_optimal_components(self, X: np.ndarray, n_range: range = range(2, 11)) -> dict:
        """
        Evaluates component count based on BIC, AIC, and Silhouette Score.
        """
        results = []
        best_n = self.n_components
        best_bic = float('inf')
        
        for n in n_range:
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            labels = gmm.fit_predict(X)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            results.append({
                'n_components': n,
                'bic': float(bic),
                'aic': float(aic),
                'silhouette_score': float(sil),
                'davies_bouldin': float(db),
                'calinski_harabasz': float(ch)
            })
            
            if bic < best_bic:
                best_bic = bic
                best_n = n

        return {
            'best_n': best_n,
            'best_bic': best_bic,
            'grid_search': results
        }

    def fit(self, X: np.ndarray):
        """
        Fits GMM model on dataset X.
        """
        self.model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)
        self.means_ = self.model.means_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts hard cluster label for input matrix X.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts soft cluster probabilities for input matrix X.
        """
        return self.model.predict_proba(X)
