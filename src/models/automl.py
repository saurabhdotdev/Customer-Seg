import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.models.kmeans_model import KMeansSegmentationModel
from src.models.gmm_model import GMMSegmentationModel
from src.models.hierarchical_model import HierarchicalSegmentationModel
from src.models.dbscan_model import DBSCANSegmentationModel

class AutoMLModelSelector:
    """
    Auto-ML Production Model Selection & Benchmarking Engine.
    Evaluates K-Means, GMM, DBSCAN, and Hierarchical Clustering across Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
    """

    @classmethod
    def run_automl_benchmark(cls, X_matrix: np.ndarray, optimal_k: int = 4) -> dict:
        models = {
            "K-Means Clustering": KMeansSegmentationModel(n_clusters=optimal_k, random_state=42),
            "Gaussian Mixture Model (GMM)": GMMSegmentationModel(n_components=optimal_k, random_state=42),
            "Hierarchical Agglomerative": HierarchicalSegmentationModel(n_clusters=optimal_k, linkage="ward"),
            "DBSCAN (Density-Based)": DBSCANSegmentationModel(eps=1.5, min_samples=5)
        }

        results = []
        best_score = -1.0
        winner_name = "K-Means Clustering"

        for name, model in models.items():
            try:
                model.fit(X_matrix)
                labels = model.labels_

                # Exclude noise points (-1) for valid cluster metric computation
                valid_mask = labels != -1
                n_clusters_found = len(set(labels[valid_mask]))

                if n_clusters_found > 1 and valid_mask.sum() > n_clusters_found:
                    sil = float(silhouette_score(X_matrix[valid_mask], labels[valid_mask]))
                    ch = float(calinski_harabasz_score(X_matrix[valid_mask], labels[valid_mask]))
                    db = float(davies_bouldin_score(X_matrix[valid_mask], labels[valid_mask]))
                else:
                    sil, ch, db = 0.0, 0.0, 99.0

                # Score heuristic: 0.6 * Silhouette + 0.4 * Normalized Calinski-Harabasz
                composite_score = round(sil * 0.7 + (min(1.0, ch / 5000.0)) * 0.3, 4)

                if composite_score > best_score:
                    best_score = composite_score
                    winner_name = name

                results.append({
                    "model_name": name,
                    "silhouette_score": round(sil, 4),
                    "calinski_harabasz_score": round(ch, 1),
                    "davies_bouldin_score": round(db, 4),
                    "composite_score": composite_score,
                    "n_clusters": n_clusters_found,
                    "is_recommended": False
                })
            except Exception as exc:
                print(f"AutoML evaluation failed for {name}:", exc)

        for r in results:
            if r["model_name"] == winner_name:
                r["is_recommended"] = True

        return {
            "winner_model": winner_name,
            "best_composite_score": best_score,
            "leaderboard": sorted(results, key=lambda x: x["composite_score"], reverse=True),
            "recommendation_reason": f"Selected '{winner_name}' based on optimal cluster separation (Silhouette: {best_score})."
        }
