import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ModelBenchmarkEvaluator:
    """
    Unified evaluation benchmark for evaluating and comparing clustering algorithms:
    K-Means, DBSCAN, Hierarchical Agglomerative Clustering, and GMM.
    """
    @staticmethod
    def evaluate_all(X: np.ndarray, models_dict: dict) -> pd.DataFrame:
        """
        Calculates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index,
        and cluster distributions across a dictionary of trained models.
        """
        benchmark_records = []
        
        for name, model in models_dict.items():
            labels = model.labels_
            unique_labels = set(labels)
            n_clusters = len(unique_labels - {-1})
            n_noise = list(labels).count(-1) if -1 in labels else 0
            
            # Filter noise points for metric evaluation if DBSCAN
            valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
            
            if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
                sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
                db = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
                ch = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
            else:
                sil, db, ch = -1.0, 999.0, 0.0

            benchmark_records.append({
                'Algorithm': str(name),
                'Clusters': int(n_clusters),
                'Outliers/Noise': int(n_noise),
                'Silhouette_Score': float(round(sil, 4)),
                'Davies_Bouldin_Index': float(round(db, 4)),
                'Calinski_Harabasz_Index': float(round(ch, 2))
            })


        df_results = pd.DataFrame(benchmark_records)
        df_results = df_results.sort_values(by='Silhouette_Score', ascending=False).reset_index(drop=True)
        return df_results
