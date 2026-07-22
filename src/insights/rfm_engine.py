import pandas as pd
import numpy as np
from typing import Dict, Any, List

class RFMAnalyticsEngine:
    """
    Computes Recency, Frequency, and Monetary (RFM) quintile scores (1-5),
    segments customers into humanized RFM categories, and generates human-readable 5x5 heatmap matrix.
    """

    CELL_TITLES = {
        (5, 5): {"title": "VIP Champions", "action": "Give exclusive VIP lounge access & early product releases.", "color": "#10B981"},
        (5, 4): {"title": "Loyal Superstars", "action": "Reward loyalty with point boosts & personalized perks.", "color": "#10B981"},
        (5, 3): {"title": "Active Regulars", "action": "Cross-sell new arrivals & encourage higher order value.", "color": "#3B82F6"},
        (5, 2): {"title": "Promising Shoppers", "action": "Offer free shipping to turn into frequent buyers.", "color": "#3B82F6"},
        (5, 1): {"title": "New First-Timers", "action": "Send welcome email series with onboarding discounts.", "color": "#06B6D4"},

        (4, 5): {"title": "High-Value Frequent", "action": "Offer premium concierge support & referral rewards.", "color": "#10B981"},
        (4, 4): {"title": "Loyal Customers", "action": "Recommend complementary product bundles.", "color": "#3B82F6"},
        (4, 3): {"title": "Potential Loyalists", "action": "Engage with loyalty program invitations.", "color": "#8B5CF6"},
        (4, 2): {"title": "Recent Regulars", "action": "Send product recommendations based on past purchase.", "color": "#8B5CF6"},
        (4, 1): {"title": "Recent One-Timers", "action": "Send 10% discount on second purchase.", "color": "#06B6D4"},

        (3, 5): {"title": "Consistent Spenders", "action": "Engage before recency slips further.", "color": "#F59E0B"},
        (3, 4): {"title": "Needs Attention", "action": "Send targeted email push for recent store offers.", "color": "#F59E0B"},
        (3, 3): {"title": "Average Shoppers", "action": "Offer seasonal discount vouchers.", "color": "#F59E0B"},
        (3, 2): {"title": "Cooling Down", "action": "Re-engage with limited-time weekend sale alerts.", "color": "#F97316"},
        (3, 1): {"title": "Casual Buyers", "action": "Highlight trending items to spark re-interest.", "color": "#F97316"},

        (2, 5): {"title": "Can't Lose Them!", "action": "High historical value! Call or send VIP win-back gift.", "color": "#EF4444"},
        (2, 4): {"title": "At-Risk VIPs", "action": "Send urgent 20% discount & personalized message.", "color": "#EF4444"},
        (2, 3): {"title": "About To Sleep", "action": "Send strong reactivation discount coupon immediately.", "color": "#F97316"},
        (2, 2): {"title": "Slipping Away", "action": "Send re-engagement survey with promo code incentive.", "color": "#F97316"},
        (2, 1): {"title": "Dormant Buyers", "action": "Run low-cost automated email retargeting.", "color": "#64748B"},

        (1, 5): {"title": "Critical At-Risk", "action": "Aggressive win-back campaign with high discount.", "color": "#DC2626"},
        (1, 4): {"title": "Lapsed High Spenders", "action": "Re-engage with major promotional incentive.", "color": "#DC2626"},
        (1, 3): {"title": "Slipped Away", "action": "Include in monthly newsletter broadcast.", "color": "#64748B"},
        (1, 2): {"title": "Hibernating", "action": "Standard retargeting ads.", "color": "#64748B"},
        (1, 1): {"title": "Lost Customers", "action": "Don't spend heavy ad budget; low recovery priority.", "color": "#475569"}
    }

    @classmethod
    def calculate_rfm_scores(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_rfm = df.copy()
        df_rfm['R_Score'] = pd.qcut(df_rfm['Recency_Days'].rank(method='first', ascending=False), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency_Orders'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary_Spend'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        return df_rfm

    @classmethod
    def generate_rfm_analysis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        df_rfm = cls.calculate_rfm_scores(df)

        heatmap_matrix = []
        for r in range(5, 0, -1):
            row_cells = []
            for f in range(1, 6):
                cell_df = df_rfm[(df_rfm['R_Score'] == r) & (df_rfm['F_Score'] == f)]
                count = len(cell_df)
                avg_spend = round(float(cell_df['Monetary_Spend'].mean()), 2) if count > 0 else 0.0
                meta = cls.CELL_TITLES.get((r, f), {"title": f"R{r}/F{f}", "action": "N/A", "color": "#3B82F6"})

                row_cells.append({
                    "r_score": r,
                    "f_score": f,
                    "count": count,
                    "avg_spend": avg_spend,
                    "title": meta["title"],
                    "action": meta["action"],
                    "color": meta["color"]
                })
            heatmap_matrix.append(row_cells)

        def assign_cohort(r, f):
            if r >= 4 and f >= 4:
                return ("👑 Champions & VIPs", "#10B981")
            elif r >= 4 and f <= 2:
                return ("🆕 New First-Timers", "#06B6D4")
            elif r >= 3 and f >= 3:
                return ("⚡ Active Regulars", "#3B82F6")
            elif r <= 2 and f >= 4:
                return ("🚨 At Risk (High Value)", "#EF4444")
            elif r == 3 and f <= 2:
                return ("⚠️ Cooling Down", "#F59E0B")
            else:
                return ("💤 Lost & Hibernating", "#64748B")

        cohort_counts = {}
        cohort_colors = {}
        total = len(df_rfm)

        for row in heatmap_matrix:
            for cell in row:
                c_name, c_color = assign_cohort(cell["r_score"], cell["f_score"])
                cohort_counts[c_name] = cohort_counts.get(c_name, 0) + cell["count"]
                cohort_colors[c_name] = c_color

        cohort_summary = []
        for name, cnt in cohort_counts.items():
            pct = round((cnt / total) * 100, 1) if total > 0 else 0
            cohort_summary.append({
                "cohort_name": name,
                "count": cnt,
                "percentage": pct,
                "color": cohort_colors[name]
            })

        return {
            "total_customers": len(df_rfm),
            "heatmap_matrix": heatmap_matrix,
            "cohort_summary": cohort_summary
        }
