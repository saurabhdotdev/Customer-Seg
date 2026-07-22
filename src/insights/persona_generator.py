import numpy as np
import pandas as pd

class CustomerPersonaGenerator:
    """
    Analyzes cluster centroids and feature distributions to generate human-readable
    customer personas, business insights, financial metrics, and tailored marketing playbooks.
    """
    
    PERSONA_CATALOG = {
        'CHAMPION': {
            'title': '🌟 VIP Champions',
            'tagline': 'High Spend, High Frequency & Exceptional Loyalty',
            'description': 'Your top-tier customers who buy frequently, spend the most, and actively engage with your brand.',
            'strategy': 'Provide VIP concierge support, early access to new product lines, exclusive rewards, and brand ambassador opportunities.',
            'color': '#10B981', # Emerald
            'icon': 'crown'
        },
        'LOYALIST': {
            'title': '💎 Steady Loyalists',
            'tagline': 'Consistent Purchasers & High Lifetime Value Potential',
            'description': 'Regular customers with reliable ordering patterns and solid order value.',
            'strategy': 'Implement cross-sell & up-sell recommendations, tier-upgrade incentives, and loyalty points accelerators.',
            'color': '#3B82F6', # Blue
            'icon': 'shield-check'
        },
        'AT_RISK': {
            'title': '⚠️ At-Risk / Hibernating',
            'tagline': 'High Past Value, but Prolonged Inactivity',
            'description': 'Previously valuable customers who have not made a purchase recently and show high churn risk.',
            'strategy': 'Launch win-back email drip campaigns, personalized re-engagement discounts, and feedback surveys.',
            'color': '#EF4444', # Red
            'icon': 'alert-triangle'
        },
        'BARGAIN': {
            'title': '🏷️ Bargain Hunters',
            'tagline': 'Discount-Driven & Price Sensitive',
            'description': 'Customers who buy primarily during promotional sales and clearance events with high discount usage.',
            'strategy': 'Target with seasonal clearance alerts, bundle discount offers, and flash sales without eroding margin.',
            'color': '#F59E0B', # Amber
            'icon': 'tag'
        },
        'NEW_BUYER': {
            'title': '🌱 New Potential Loyalists',
            'tagline': 'Recent First-Time Buyers with Growth Trajectory',
            'description': 'Recent buyers with low transaction count but promising initial engagement metrics.',
            'strategy': 'Deliver engaging onboarding series, product education guides, and second-purchase discount incentives.',
            'color': '#8B5CF6', # Purple
            'icon': 'sparkles'
        },
        'GENERAL': {
            'title': '📊 Standard Segment',
            'tagline': 'Moderate Customer Engagement Profile',
            'description': 'Standard customer group showing balanced behavioral metrics across all categories.',
            'strategy': 'Apply standard marketing newsletters, general promotions, and product recommendations.',
            'color': '#6B7280', # Gray
            'icon': 'user'
        }
    }

    @classmethod
    def analyze_clusters(cls, df: pd.DataFrame, cluster_col: str = 'Cluster') -> dict:
        """
        Calculates persona profiles, financial summaries, and actionable recommendations.
        """
        total_customers = len(df)
        total_revenue = float(df['Monetary_Spend'].sum())
        
        clusters_summary = []
        cluster_labels_map = {}
        
        for cluster_id in sorted(df[cluster_col].unique()):
            c_df = df[df[cluster_col] == cluster_id]
            c_count = len(c_df)
            c_pct = round((c_count / total_customers) * 100.0, 2)
            c_rev = float(c_df['Monetary_Spend'].sum())
            c_rev_pct = round((c_rev / total_revenue) * 100.0, 2)
            
            # Means
            avg_recency = float(c_df['Recency_Days'].mean())
            avg_freq = float(c_df['Frequency_Orders'].mean())
            avg_monetary = float(c_df['Monetary_Spend'].mean())
            avg_aov = float(c_df['Avg_Order_Value'].mean())
            avg_engagement = float(c_df['Engagement_Score'].mean())
            avg_discount = float(c_df['Discount_Ratio'].mean())
            avg_churn = float(c_df['Churn_Risk_Index'].mean())
            
            # Match persona based on centroid characteristics
            if cluster_id == -1:
                key = 'GENERAL'
                persona_info = {
                    'title': '🚨 Outliers / Noise',
                    'tagline': 'Irregular or Anomalous Behavioral Patterns',
                    'description': 'Customers with extreme values or unusual purchasing frequencies flagged by DBSCAN.',
                    'strategy': 'Manually audit transaction logs for fraud prevention or edge-case handling.',
                    'color': '#64748B',
                    'icon': 'ghost'
                }
            elif avg_monetary > df['Monetary_Spend'].quantile(0.70) and avg_freq > df['Frequency_Orders'].quantile(0.65) and avg_recency < df['Recency_Days'].quantile(0.40):
                key = 'CHAMPION'
                persona_info = cls.PERSONA_CATALOG[key]
            elif avg_recency > df['Recency_Days'].quantile(0.60) and avg_monetary > df['Monetary_Spend'].median():
                key = 'AT_RISK'
                persona_info = cls.PERSONA_CATALOG[key]
            elif avg_discount > 0.55:
                key = 'BARGAIN'
                persona_info = cls.PERSONA_CATALOG[key]
            elif avg_freq <= 4 and avg_recency < 40:
                key = 'NEW_BUYER'
                persona_info = cls.PERSONA_CATALOG[key]
            elif avg_freq >= df['Frequency_Orders'].median():
                key = 'LOYALIST'
                persona_info = cls.PERSONA_CATALOG[key]
            else:
                key = 'GENERAL'
                persona_info = cls.PERSONA_CATALOG[key]

            cluster_labels_map[str(int(cluster_id))] = persona_info['title']
            
            clusters_summary.append({
                'cluster_id': int(cluster_id),
                'persona_key': key,
                'persona_title': persona_info['title'],
                'tagline': persona_info['tagline'],
                'description': persona_info['description'],
                'strategy': persona_info['strategy'],
                'color': persona_info['color'],
                'icon': persona_info['icon'],
                'metrics': {
                    'customer_count': int(c_count),
                    'customer_percentage': float(c_pct),
                    'total_revenue': float(round(c_rev, 2)),
                    'revenue_percentage': float(c_rev_pct),
                    'avg_recency_days': float(round(avg_recency, 1)),
                    'avg_frequency_orders': float(round(avg_freq, 1)),
                    'avg_monetary_spend': float(round(avg_monetary, 2)),
                    'avg_order_value': float(round(avg_aov, 2)),
                    'avg_engagement_score': float(round(avg_engagement, 1)),
                    'avg_discount_ratio': float(round(avg_discount, 2)),
                    'avg_churn_risk': float(round(avg_churn, 3))
                }
            })
            
        return {
            'total_customers': int(total_customers),
            'total_revenue': float(round(total_revenue, 2)),
            'cluster_labels_map': cluster_labels_map,
            'clusters': clusters_summary
        }

