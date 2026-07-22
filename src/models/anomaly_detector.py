import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class CustomerAnomalyDetector:
    """
    IsolationForest Anomaly & Fraud Detection Engine for Customer Analytics.
    Detects extreme whale spenders, abnormal return behavior, high support ticket frequency, and suspicious anomalies.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
            n_jobs=-1
        )
        self.feature_columns = [
            "Recency_Days", "Frequency_Orders", "Monetary_Spend",
            "Category_Diversity", "Engagement_Score", "Support_Tickets",
            "Discount_Ratio", "Return_Rate"
        ]

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        missing_cols = [c for c in self.feature_columns if c not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for anomaly detection: {missing_cols}")

        X = df_clean[self.feature_columns].copy()
        
        # Fit IsolationForest
        predictions = self.model.fit_predict(X)  # 1 for inliers, -1 for outliers
        scores = self.model.score_samples(X)    # Negative anomaly score

        # Normalize score between 0.0 (normal) and 1.0 (extreme anomaly)
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            norm_scores = 1.0 - ((scores - min_score) / (max_score - min_score))
        else:
            norm_scores = np.zeros(len(scores))

        df_clean["Is_Anomaly"] = (predictions == -1)
        df_clean["Anomaly_Score"] = np.round(norm_scores, 4)
        df_clean["Anomaly_Type"] = df_clean.apply(self._categorize_anomaly, axis=1)

        return df_clean

    def predict_single(self, input_dict: dict, model_path: str = None) -> dict:
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
            except Exception:
                pass

        row_df = pd.DataFrame([input_dict])
        for col in self.feature_columns:
            if col not in row_df.columns:
                row_df[col] = 0.0

        X = row_df[self.feature_columns]

        from sklearn.utils.validation import check_is_fitted, NotFittedError
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            from src.data.generator import generate_customer_dataset
            synth_df = generate_customer_dataset(n_samples=200, random_state=42)
            self.fit_predict(synth_df)

        pred = self.model.predict(X)[0]
        score = self.model.score_samples(X)[0]

        is_anomaly = bool(pred == -1)
        severity = round(float(np.clip(-score, 0.0, 1.0)), 4)
        category = self._categorize_anomaly(row_df.iloc[0], forced_anomaly=is_anomaly)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": severity,
            "anomaly_type": category
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)

    @staticmethod
    def _categorize_anomaly(row, forced_anomaly: bool = None) -> str:
        is_anom = forced_anomaly if forced_anomaly is not None else row.get("Is_Anomaly", False)
        if not is_anom:
            return "Normal Pattern"

        spend = row.get("Monetary_Spend", 0)
        freq = row.get("Frequency_Orders", 0)
        tickets = row.get("Support_Tickets", 0)
        return_rate = row.get("Return_Rate", 0)
        recency = row.get("Recency_Days", 0)

        if spend > 15000 or (spend > 8000 and freq > 40):
            return "VIP Whale Outlier"
        elif tickets >= 5 or (tickets >= 3 and return_rate > 0.3):
            return "High Friction / Fraud Suspect"
        elif return_rate > 0.35:
            return "Abnormal Return Behavior"
        elif recency > 180 and spend > 5000:
            return "Dormant High Value Customer"
        else:
            return "Behavioral Outlier"
