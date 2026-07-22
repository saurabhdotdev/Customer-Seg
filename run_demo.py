import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from backend.config import DATA_RAW_PATH
from src.data.generator import generate_customer_dataset
from src.pipeline.training import train_customer_segmentation_pipeline


def run_pipeline(n_samples: int = 50000):
    start_time = time.time()
    print("=" * 60)
    print("LARGE-SCALE CUSTOMER INTELLIGENCE ML PIPELINE")
    print("=" * 60)

    print(f"\n[Step 1/3] Generating {n_samples:,} synthetic customer records...")
    df_raw = generate_customer_dataset(n_samples=n_samples, random_state=42, output_path=DATA_RAW_PATH)
    gen_time = time.time() - start_time
    print(f" -> Data Generation completed in {gen_time:.2f} seconds ({len(df_raw)/gen_time:.0f} records/sec)")

    print("\n[Step 2/3] Training segmentation, anomaly, PCA, and intelligence layers...")
    train_start = time.time()
    metadata = train_customer_segmentation_pipeline(df_raw, data_source=f"synthetic_demo_{n_samples}")
    train_time = time.time() - train_start

    print("\n[Step 3/3] Training summary & performance benchmarks")
    print(f" -> Total Records Processed: {metadata['total_samples']:,}")
    print(f" -> Production Model: {metadata['production_model']}")
    print(f" -> Anomaly Model: {metadata['anomaly_model']}")
    print(f" -> Optimal K: {metadata['optimal_k']}")
    print(f" -> Stability ARI: {metadata['stability_report']['mean_adjusted_rand_index']}")
    print(f" -> DBSCAN Anomalies: {metadata['anomaly_summary']['count']}")
    print(f" -> Model Training Duration: {train_time:.2f} seconds")
    print(f" -> Total Pipeline Duration: {time.time() - start_time:.2f} seconds")

    print("\n" + "=" * 60)
    print("[SUCCESS] LARGE-SCALE ML PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    sample_size = 50000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        sample_size = int(sys.argv[1])
        
    run_pipeline(n_samples=sample_size)

    if "--no-server" not in sys.argv:
        print("\n[SERVER] Starting FastAPI Web Server at http://127.0.0.1:8000 ...")
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
