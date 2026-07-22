import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data.generator import generate_customer_dataset
from src.data.preprocessor import CustomerDataPreprocessor
from src.models.kmeans_model import KMeansSegmentationModel
from src.models.dbscan_model import DBSCANSegmentationModel
from src.models.hierarchical_model import HierarchicalSegmentationModel
from src.models.gmm_model import GMMSegmentationModel
from src.models.evaluator import ModelBenchmarkEvaluator
from src.visualization.dimensionality import DimensionalityReducer
from src.insights.persona_generator import CustomerPersonaGenerator
from backend.config import (
    DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_SEGMENTS_PATH, 
    SCALER_PATH, PCA_PATH, METADATA_PATH
)

def run_pipeline():
    print("=" * 60)
    print("HIGH-YIELD CUSTOMER SEGMENTATION ML PIPELINE")
    print("=" * 60)
    
    # 1. Dataset Generation
    print("\n[Step 1/6] Generating 5,000 synthetic customer records...")
    df_raw = generate_customer_dataset(n_samples=5000, random_state=42, output_path=DATA_RAW_PATH)
    
    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/6] Calculating RFM scores, scaling & log transforms...")
    preprocessor = CustomerDataPreprocessor(scaler_save_path=SCALER_PATH)
    X_matrix, df_processed = preprocessor.fit_transform(df_raw)
    
    os.makedirs(os.path.dirname(DATA_PROCESSED_PATH), exist_ok=True)
    df_processed.to_csv(DATA_PROCESSED_PATH, index=False)
    
    # 3. Model Training Suite (K-Means, DBSCAN, HAC, GMM)
    print("\n[Step 3/6] Training & Tuning 4 Clustering Algorithms...")
    
    # K-Means
    kmeans = KMeansSegmentationModel()
    k_search = kmeans.find_optimal_k(X_matrix, range(2, 10))
    optimal_k = k_search['best_k']
    kmeans.n_clusters = optimal_k
    kmeans.fit(X_matrix)
    print(f" -> K-Means Optimal K: {optimal_k} (Silhouette: {k_search['best_silhouette']:.4f})")
    
    # DBSCAN
    dbscan = DBSCANSegmentationModel()
    db_search = dbscan.tune_hyperparameters(X_matrix)
    dbscan.eps = db_search['best_params']['eps']
    dbscan.min_samples = db_search['best_params']['min_samples']
    dbscan.fit(X_matrix)
    print(f" -> DBSCAN Best Epsilon: {dbscan.eps}, MinSamples: {dbscan.min_samples}")
    
    # Hierarchical Clustering
    hac = HierarchicalSegmentationModel(n_clusters=optimal_k, linkage='ward')
    hac.fit(X_matrix)
    print(f" -> Hierarchical Agglomerative Clustering (K={optimal_k})")
    
    # GMM
    gmm = GMMSegmentationModel(n_components=optimal_k)
    gmm.fit(X_matrix)
    print(f" -> Gaussian Mixture Model (Components={optimal_k})")
    
    # 4. Dimensionality Reduction (PCA)
    print("\n[Step 4/6] Computing PCA 3D Projections...")
    reducer = DimensionalityReducer(n_components_pca=3, pca_save_path=PCA_PATH)
    X_pca, pca_meta = reducer.fit_transform_pca(X_matrix)
    
    # Append labels to DataFrame
    df_processed['KMeans_Cluster'] = kmeans.labels_
    df_processed['DBSCAN_Cluster'] = dbscan.labels_
    df_processed['HAC_Cluster'] = hac.labels_
    df_processed['GMM_Cluster'] = gmm.labels_
    
    df_processed['PCA_1'] = X_pca[:, 0]
    df_processed['PCA_2'] = X_pca[:, 1]
    df_processed['PCA_3'] = X_pca[:, 2]
    
    df_processed.to_csv(DATA_SEGMENTS_PATH, index=False)
    print(f" -> Saved segment labels & PCA coordinates to {DATA_SEGMENTS_PATH}")
    
    # 5. Benchmark Evaluation
    print("\n[Step 5/6] Benchmarking Models & Generating Personas...")
    models_dict = {
        'K-Means Clustering': kmeans,
        'Gaussian Mixture Model (GMM)': gmm,
        'Hierarchical Agglomerative': hac,
        'DBSCAN (Density-Based)': dbscan
    }
    df_benchmark = ModelBenchmarkEvaluator.evaluate_all(X_matrix, models_dict)
    print("\nBenchmark Results:")
    print(df_benchmark.to_string(index=False))
    
    # 6. Save Metadata Artifacts
    personas_data = CustomerPersonaGenerator.analyze_clusters(df_processed, cluster_col='KMeans_Cluster')
    
    metadata = {
        'total_samples': len(df_processed),
        'optimal_k': int(optimal_k),
        'kmeans_centroids': kmeans.cluster_centers_.tolist(),
        'pca_metadata': pca_meta,
        'benchmark_comparison': df_benchmark.to_dict(orient='records'),
        'persona_summary': personas_data
    }
    
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" -> Saved metadata to {METADATA_PATH}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ML PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
    
    if "--no-server" not in sys.argv:
        print("\n[SERVER] Starting FastAPI Web Server at http://127.0.0.1:8000 ...")
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


