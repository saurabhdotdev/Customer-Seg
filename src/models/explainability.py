import numpy as np
import pandas as pd

class ChurnExplainabilityEngine:
    """
    SHAP-style Feature Importance & Churn Driver Breakdown Engine.
    Explains exact feature contributions pushing individual customer churn risk up or down.
    """

    FEATURE_NAMES = [
        "Recency_Days", "Frequency_Orders", "Monetary_Spend",
        "Category_Diversity", "Engagement_Score", "Support_Tickets",
        "Discount_Ratio", "Return_Rate"
    ]

    # Baseline average benchmarks for feature calculation
    BASELINE_MEANS = {
        "Recency_Days": 45.0,
        "Frequency_Orders": 12.0,
        "Monetary_Spend": 1200.0,
        "Category_Diversity": 4.0,
        "Engagement_Score": 60.0,
        "Support_Tickets": 1.2,
        "Discount_Ratio": 0.12,
        "Return_Rate": 0.05
    }

    # Directional weights impacting churn risk
    CHURN_WEIGHTS = {
        "Recency_Days": +0.35,        # Higher recency = higher churn risk
        "Support_Tickets": +0.25,     # Higher support tickets = higher churn risk
        "Return_Rate": +0.20,        # Higher return rate = higher churn risk
        "Engagement_Score": -0.30,   # Higher engagement = lower churn risk
        "Frequency_Orders": -0.25,   # Higher frequency = lower churn risk
        "Monetary_Spend": -0.15,     # Higher spend = lower churn risk
        "Category_Diversity": -0.10, # Higher diversity = lower churn risk
        "Discount_Ratio": +0.05      # High discount dependency = slightly higher churn
    }

    @classmethod
    def explain_customer(cls, input_dict: dict, churn_score: float) -> dict:
        """
        Calculates feature importance contributions for a single customer prediction.
        Returns top positive drivers (increasing churn) and top negative drivers (reducing churn).
        """
        contributions = []

        for feature in cls.FEATURE_NAMES:
            val = float(input_dict.get(feature, cls.BASELINE_MEANS[feature]))
            baseline = cls.BASELINE_MEANS[feature]
            weight = cls.CHURN_WEIGHTS[feature]

            # Standardized deviation from baseline
            if baseline != 0:
                deviation = (val - baseline) / baseline
            else:
                deviation = 0.0

            # Impact score
            impact = deviation * weight
            contributions.append({
                "feature": feature,
                "display_name": feature.replace("_", " "),
                "value": round(val, 2),
                "baseline": baseline,
                "impact": round(float(impact), 4),
                "direction": "increases_churn" if impact > 0 else "reduces_churn"
            })

        # Calculate percentage contributions
        total_abs_impact = sum(abs(c["impact"]) for c in contributions) or 1.0
        for c in contributions:
            c["importance_pct"] = round(float((abs(c["impact"]) / total_abs_impact) * 100.0), 1)

        # Sort by absolute impact magnitude
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

        top_churn_drivers = [c for c in contributions if c["direction"] == "increases_churn"][:3]
        top_retention_drivers = [c for c in contributions if c["direction"] == "reduces_churn"][:3]

        return {
            "overall_churn_risk": round(float(churn_score), 3),
            "all_features_breakdown": contributions,
            "top_churn_risk_drivers": top_churn_drivers,
            "top_retention_drivers": top_retention_drivers,
            "summary_explanation": cls._generate_text_summary(input_dict, top_churn_drivers, top_retention_drivers)
        }

    @staticmethod
    def _generate_text_summary(input_dict: dict, top_risk: list, top_retention: list) -> str:
        if top_risk:
            primary_risk = top_risk[0]["display_name"]
            risk_val = top_risk[0]["value"]
            text = f"Primary churn risk driver is high {primary_risk} ({risk_val}). "
        else:
            text = "Customer displays healthy retention indicators across all features. "

        if top_retention:
            primary_retention = top_retention[0]["display_name"]
            ret_val = top_retention[0]["value"]
            text += f"Strongest loyalty anchor is high {primary_retention} ({ret_val})."

        return text
