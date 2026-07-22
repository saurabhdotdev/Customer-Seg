import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split


class SegmentIntelligenceEngine:
    """
    Converts clustering output into decision-ready business intelligence:
    explainability, campaign actions, anomaly summaries, and stability checks.
    """

    FEATURE_COLUMNS = [
        "Recency_Days",
        "Frequency_Orders",
        "Monetary_Spend",
        "Avg_Order_Value",
        "Category_Diversity",
        "Engagement_Score",
        "Discount_Ratio",
        "Return_Rate",
        "Churn_Risk_Index",
    ]

    @classmethod
    def build_segment_explanations(cls, df: pd.DataFrame, cluster_col: str = "KMeans_Cluster") -> list[dict]:
        global_means = df[cls.FEATURE_COLUMNS].mean()
        global_stds = df[cls.FEATURE_COLUMNS].std().replace(0, 1)
        explanations = []

        for cluster_id in sorted(df[cluster_col].unique()):
            c_df = df[df[cluster_col] == cluster_id]
            c_means = c_df[cls.FEATURE_COLUMNS].mean()
            z_scores = ((c_means - global_means) / global_stds).sort_values(key=np.abs, ascending=False)

            drivers = []
            for feature, z_value in z_scores.head(4).items():
                direction = "above average" if z_value > 0 else "below average"
                drivers.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "strength": round(float(abs(z_value)), 3),
                        "cluster_average": round(float(c_means[feature]), 3),
                        "global_average": round(float(global_means[feature]), 3),
                    }
                )

            explanations.append(
                {
                    "cluster_id": int(cluster_id),
                    "sample_size": int(len(c_df)),
                    "top_drivers": drivers,
                    "plain_english": cls._plain_english_summary(drivers),
                }
            )

        return explanations

    @classmethod
    def build_campaign_recommendations(cls, df: pd.DataFrame, personas: dict) -> list[dict]:
        label_map = personas.get("cluster_labels_map", {})
        campaigns = []

        for cluster in personas.get("clusters", []):
            cluster_id = cluster["cluster_id"]
            metrics = cluster["metrics"]
            c_df = df[df["KMeans_Cluster"] == cluster_id]

            expected_revenue_at_risk = (
                float(c_df["Monetary_Spend"].sum()) * float(c_df["Churn_Risk_Index"].mean())
            )
            priority_score = (
                0.45 * metrics["revenue_percentage"]
                + 35.0 * metrics["avg_churn_risk"]
                + 0.20 * metrics["customer_percentage"]
            )

            campaigns.append(
                {
                    "cluster_id": int(cluster_id),
                    "persona_title": label_map.get(str(cluster_id), cluster["persona_title"]),
                    "priority_score": round(float(priority_score), 2),
                    "expected_revenue_at_risk": round(float(expected_revenue_at_risk), 2),
                    "recommended_campaign": cls._campaign_for_persona(cluster["persona_key"]),
                    "success_metric": cls._success_metric_for_persona(cluster["persona_key"]),
                    "estimated_audience": int(metrics["customer_count"]),
                }
            )

        return sorted(campaigns, key=lambda x: x["priority_score"], reverse=True)

    @classmethod
    def summarize_anomalies(cls, df: pd.DataFrame, top_n: int = 25) -> dict:
        if "DBSCAN_Cluster" not in df.columns:
            return {"count": 0, "percentage": 0.0, "customers": []}

        anomalies = df[df["DBSCAN_Cluster"] == -1].copy()
        if anomalies.empty:
            return {"count": 0, "percentage": 0.0, "customers": []}

        anomalies["Anomaly_Severity"] = (
            anomalies["Monetary_Spend"].rank(pct=True) * 0.35
            + anomalies["Return_Rate"].rank(pct=True) * 0.25
            + anomalies["Discount_Ratio"].rank(pct=True) * 0.20
            + anomalies["Churn_Risk_Index"].rank(pct=True) * 0.20
        )
        top = anomalies.sort_values("Anomaly_Severity", ascending=False).head(top_n)

        return {
            "count": int(len(anomalies)),
            "percentage": round(float(len(anomalies) / len(df) * 100), 2),
            "interpretation": "DBSCAN is used as a risk and anomaly layer, while K-Means remains the production segmentation model.",
            "customers": [
                {
                    "customer_id": row["Customer_ID"],
                    "severity": round(float(row["Anomaly_Severity"]), 3),
                    "monetary_spend": round(float(row["Monetary_Spend"]), 2),
                    "return_rate": round(float(row["Return_Rate"]), 3),
                    "discount_ratio": round(float(row["Discount_Ratio"]), 3),
                    "churn_risk_index": round(float(row["Churn_Risk_Index"]), 3),
                }
                for _, row in top.iterrows()
            ],
        }

    @staticmethod
    def estimate_kmeans_stability(X: np.ndarray, n_clusters: int, random_state: int = 42, runs: int = 5) -> dict:
        scores = []
        eval_X = X
        if len(X) > 10000:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), size=10000, replace=False)
            eval_X = X[idx]

        for run in range(runs):
            train_a, train_b = train_test_split(
                eval_X,
                test_size=0.30,
                random_state=random_state + run,
                shuffle=True,
            )
            model_a = KMeans(n_clusters=n_clusters, random_state=random_state + run, n_init=3).fit(train_a)
            model_b = KMeans(n_clusters=n_clusters, random_state=random_state + 100 + run, n_init=3).fit(train_b)

            labels_a = model_a.predict(eval_X)
            labels_b = model_b.predict(eval_X)
            scores.append(float(adjusted_rand_score(labels_a, labels_b)))

        return {
            "method": "Adjusted Rand Index across repeated train/test resamples",
            "runs": int(runs),
            "mean_adjusted_rand_index": round(float(np.mean(scores)), 4),
            "min_adjusted_rand_index": round(float(np.min(scores)), 4),
            "max_adjusted_rand_index": round(float(np.max(scores)), 4),
            "interpretation": "Scores closer to 1.0 indicate stable, reproducible customer segments.",
        }

    @staticmethod
    def _plain_english_summary(drivers: list[dict]) -> str:
        phrases = [
            f"{driver['feature'].replace('_', ' ')} is {driver['direction']}"
            for driver in drivers[:3]
        ]
        return "This segment is mainly defined by " + ", ".join(phrases) + "."

    @staticmethod
    def _campaign_for_persona(persona_key: str) -> str:
        campaigns = {
            "CHAMPION": "VIP retention program with early access, premium bundles, and referral incentives.",
            "LOYALIST": "Cross-sell and loyalty-tier upgrade journey based on preferred category behavior.",
            "AT_RISK": "Win-back sequence with personalized discount, feedback survey, and support outreach.",
            "BARGAIN": "Margin-aware flash sale targeting with bundles instead of blanket discounts.",
            "NEW_BUYER": "Second-purchase onboarding journey with education, recommendations, and welcome credit.",
            "GENERAL": "Baseline lifecycle newsletter with product recommendations and seasonal offers.",
        }
        return campaigns.get(persona_key, campaigns["GENERAL"])

    @staticmethod
    def _success_metric_for_persona(persona_key: str) -> str:
        metrics = {
            "CHAMPION": "Repeat purchase rate and referral revenue",
            "LOYALIST": "Average order value lift",
            "AT_RISK": "Reactivation rate and retained revenue",
            "BARGAIN": "Gross margin per campaign recipient",
            "NEW_BUYER": "Second purchase conversion rate",
            "GENERAL": "Click-through rate and revenue per send",
        }
        return metrics.get(persona_key, metrics["GENERAL"])
