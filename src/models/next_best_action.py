"""
Next-Best-Action (NBA) & Personalization Recommendation Engine
Evaluates customer RFM metrics, Churn Risk Score, LTV expectations, Anomaly status,
and Persona Profiles to calculate optimal retention/upsell actions, expected conversion rates,
offer costs, and net ROI lift.
"""

from typing import Dict, Any, List
import numpy as np


class NextBestActionEngine:
    """
    Enterprise Next-Best-Action Scoring Engine.
    Generates multi-factor prioritized recommendations for customer retention,
    upsell, cross-sell, and engagement.
    """

    ACTIONS_CATALOG = {
        "VIP_CONCIERGE": {
            "title": "VIP Concierge & Exclusive Beta Access",
            "category": "Retention & Loyalty",
            "offer_cost": 25.0,
            "base_conversion": 0.42,
            "ltv_lift_multiplier": 1.25,
            "badge_color": "emerald",
            "icon": "💎",
            "description": "Personal account manager outreach with priority support and invitation to exclusive product preview.",
        },
        "ANNUAL_UPGRADE_DISCOUNT": {
            "title": "20% Off Annual Plan Upgrade",
            "category": "LTV Expansion",
            "offer_cost": 40.0,
            "base_conversion": 0.35,
            "ltv_lift_multiplier": 1.40,
            "badge_color": "indigo",
            "icon": "⭐",
            "description": "Offer locked-in 20% annual discount to transition high-frequency monthly buyers to high-retention annual contracts.",
        },
        "WINBACK_CREDIT": {
            "title": "$25 Immediate Win-Back Wallet Credit",
            "category": "Churn Rescue",
            "offer_cost": 25.0,
            "base_conversion": 0.28,
            "ltv_lift_multiplier": 1.15,
            "badge_color": "rose",
            "icon": "⚠️",
            "description": "High-urgency targeted wallet credit with 7-day expiration to re-engage lapsed customers showing elevated churn risk.",
        },
        "CROSS_SELL_CATEGORY": {
            "title": "Personalized Category Expansion Bundle",
            "category": "Cross-Sell",
            "offer_cost": 15.0,
            "base_conversion": 0.38,
            "ltv_lift_multiplier": 1.20,
            "badge_color": "sky",
            "icon": "🛒",
            "description": "15% off recommendation bundle focusing on unexplored product categories to increase purchase diversity.",
        },
        "FREE_SHIPPING_INCENTIVE": {
            "title": "Free Shipping on Next 3 Orders",
            "category": "Re-engagement",
            "offer_cost": 10.0,
            "base_conversion": 0.48,
            "ltv_lift_multiplier": 1.10,
            "badge_color": "amber",
            "icon": "🚚",
            "description": "Zero-friction incentive tailored for price-sensitive bargain hunters to lower checkout abandonment.",
        },
        "ONBOARDING_VIP_GUIDE": {
            "title": "Guided Product Onboarding & $10 Welcome Bonus",
            "category": "Onboarding",
            "offer_cost": 10.0,
            "base_conversion": 0.55,
            "ltv_lift_multiplier": 1.18,
            "badge_color": "teal",
            "icon": "🌱",
            "description": "Interactive onboarding tour and $10 credit to accelerate second order velocity for new buyers.",
        },
    }

    def __init__(self):
        pass

    def evaluate_customer(
        self,
        customer_data: Dict[str, Any],
        churn_risk_score: float = 0.3,
        predicted_ltv_12m: float = 1200.0,
        is_anomaly: bool = False,
        persona_key: str = "CHAMPION",
    ) -> Dict[str, Any]:
        """
        Calculates prioritized Next-Best-Actions for a given customer profile.
        """
        recency = customer_data.get("Recency_Days", 30)
        frequency = customer_data.get("Frequency_Orders", 5)
        monetary = customer_data.get("Monetary_Spend", 500.0)
        category_diversity = customer_data.get("Category_Diversity", 3)
        discount_ratio = customer_data.get("Discount_Ratio", 0.2)
        engagement = customer_data.get("Engagement_Score", 70.0)

        recommendations = []

        for action_id, meta in self.ACTIONS_CATALOG.items():
            score = 50.0
            reasons = []

            if action_id == "WINBACK_CREDIT":
                if churn_risk_score >= 0.6:
                    score += 40.0
                    reasons.append(f"High Churn Risk ({churn_risk_score*100:.1f}%) detected")
                if recency > 90:
                    score += 25.0
                    reasons.append(f"High Recency ({recency} days dormant)")
                if is_anomaly:
                    score += 15.0
                    reasons.append("Flagged as potential dormant anomaly")

            elif action_id == "VIP_CONCIERGE":
                if persona_key == "CHAMPION" or monetary > 2000.0:
                    score += 45.0
                    reasons.append(f"High Monetary Value (${monetary:,.2f})")
                if frequency >= 10:
                    score += 20.0
                    reasons.append(f"High Purchase Frequency ({frequency} orders)")
                if churn_risk_score < 0.3:
                    score += 15.0
                    reasons.append("Low Churn Risk / Highly Loyal")

            elif action_id == "ANNUAL_UPGRADE_DISCOUNT":
                if frequency >= 6 and recency <= 45:
                    score += 35.0
                    reasons.append("Consistent repeat order pattern")
                if predicted_ltv_12m > 1500.0:
                    score += 25.0
                    reasons.append(f"Strong 12m LTV potential (${predicted_ltv_12m:,.2f})")

            elif action_id == "CROSS_SELL_CATEGORY":
                if category_diversity <= 2:
                    score += 35.0
                    reasons.append(f"Low Category Diversity ({category_diversity} categories)")
                if frequency >= 3:
                    score += 20.0
                    reasons.append("Active buyer ready for cross-sell")

            elif action_id == "FREE_SHIPPING_INCENTIVE":
                if discount_ratio >= 0.30 or persona_key == "BARGAIN":
                    score += 40.0
                    reasons.append(f"High Discount Sensitivity ({discount_ratio*100:.0f}%)")
                if recency > 45:
                    score += 15.0
                    reasons.append("Moderate recency delay")

            elif action_id == "ONBOARDING_VIP_GUIDE":
                if frequency <= 2 or persona_key == "NEW_BUYER":
                    score += 45.0
                    reasons.append(f"New Customer Profile ({frequency} lifetime orders)")
                if engagement < 50.0:
                    score += 20.0
                    reasons.append(f"Low Engagement Score ({engagement:.1f})")

            # Final metrics calculation
            offer_cost = meta["offer_cost"]
            base_conv = meta["base_conversion"]
            # Adjust conversion based on churn/engagement factor
            adjusted_conv = min(0.95, max(0.05, base_conv + (engagement - 50.0) / 200.0 - (churn_risk_score - 0.3) / 2.0))
            expected_ltv_gain = (predicted_ltv_12m * (meta["ltv_lift_multiplier"] - 1.0)) * adjusted_conv
            net_roi_score = round(((expected_ltv_gain - offer_cost) / max(1.0, offer_cost)) * 100, 1)

            final_priority = min(100.0, max(10.0, round(score, 1)))

            recommendations.append({
                "action_id": action_id,
                "title": meta["title"],
                "category": meta["category"],
                "badge_color": meta["badge_color"],
                "icon": meta["icon"],
                "description": meta["description"],
                "priority_score": final_priority,
                "expected_conversion_rate": round(adjusted_conv * 100, 1),
                "offer_cost": offer_cost,
                "expected_ltv_gain": round(expected_ltv_gain, 2),
                "net_roi_percent": net_roi_score,
                "trigger_reasons": reasons if reasons else ["General segment recommendation profile"],
            })

        # Sort recommendations by priority score descending
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

        top_action = recommendations[0] if recommendations else None

        return {
            "total_actions_evaluated": len(recommendations),
            "top_action": top_action,
            "recommendations": recommendations,
            "evaluated_metrics": {
                "churn_risk_score": round(churn_risk_score, 3),
                "predicted_ltv_12m": round(predicted_ltv_12m, 2),
                "is_anomaly": is_anomaly,
                "persona_key": persona_key,
            },
        }
