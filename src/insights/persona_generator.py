import pandas as pd


class CustomerPersonaGenerator:
    """
    Analyzes cluster centroids and feature distributions to generate human-readable
    customer personas, business insights, financial metrics, and tailored marketing playbooks.
    """

    PERSONA_CATALOG = {
        'CHAMPION': {
            'title': 'VIP Champions',
            'tagline': 'High Spend, High Frequency & Exceptional Loyalty',
            'description': 'Your top-tier customers who buy frequently, spend the most, and actively engage with your brand.',
            'strategy': 'Provide VIP concierge support, early access to new product lines, exclusive rewards, and brand ambassador opportunities.',
            'color': '#10B981',
            'icon': 'crown'
        },
        'LOYALIST': {
            'title': 'Steady Loyalists',
            'tagline': 'Consistent Purchasers & High Lifetime Value Potential',
            'description': 'Regular customers with reliable ordering patterns and solid order value.',
            'strategy': 'Implement cross-sell and up-sell recommendations, tier-upgrade incentives, and loyalty points accelerators.',
            'color': '#3B82F6',
            'icon': 'shield-check'
        },
        'AT_RISK': {
            'title': 'At-Risk / Hibernating',
            'tagline': 'High Past Value, but Prolonged Inactivity',
            'description': 'Previously valuable customers who have not made a purchase recently and show high churn risk.',
            'strategy': 'Launch win-back email drip campaigns, personalized re-engagement discounts, and feedback surveys.',
            'color': '#EF4444',
            'icon': 'alert-triangle'
        },
        'BARGAIN': {
            'title': 'Bargain Hunters',
            'tagline': 'Discount-Driven & Price Sensitive',
            'description': 'Customers who buy primarily during promotional sales and clearance events with high discount usage.',
            'strategy': 'Target with seasonal clearance alerts, bundle discount offers, and flash sales without eroding margin.',
            'color': '#F59E0B',
            'icon': 'tag'
        },
        'NEW_BUYER': {
            'title': 'New Potential Loyalists',
            'tagline': 'Recent First-Time Buyers with Growth Trajectory',
            'description': 'Recent buyers with low transaction count but promising initial engagement metrics.',
            'strategy': 'Deliver engaging onboarding series, product education guides, and second-purchase discount incentives.',
            'color': '#8B5CF6',
            'icon': 'sparkles'
        },
        'GENERAL': {
            'title': 'Standard Segment',
            'tagline': 'Moderate Customer Engagement Profile',
            'description': 'Standard customer group showing balanced behavioral metrics across all categories.',
            'strategy': 'Apply standard marketing newsletters, general promotions, and product recommendations.',
            'color': '#6B7280',
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

            avg_recency = float(c_df['Recency_Days'].mean())
            avg_freq = float(c_df['Frequency_Orders'].mean())
            avg_monetary = float(c_df['Monetary_Spend'].mean())
            avg_aov = float(c_df['Avg_Order_Value'].mean())
            avg_engagement = float(c_df['Engagement_Score'].mean())
            avg_discount = float(c_df['Discount_Ratio'].mean())
            avg_churn = float(c_df['Churn_Risk_Index'].mean())

            if cluster_id == -1:
                key = 'GENERAL'
                persona_info = {
                    'title': 'Outliers / Noise',
                    'tagline': 'Irregular or Anomalous Behavioral Patterns',
                    'description': 'Customers with extreme values or unusual purchasing frequencies flagged by DBSCAN.',
                    'strategy': 'Manually audit transaction logs for fraud prevention or edge-case handling.',
                    'color': '#64748B',
                    'icon': 'alert-circle'
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

    @classmethod
    def generate_campaign_copy(cls, persona_key: str, persona_title: str = "") -> dict:
        """
        Generates segment-targeted email, SMS, and digital ad copy tailored for a specific persona.
        """
        catalog = {
            'CHAMPION': {
                'persona_key': 'CHAMPION',
                'persona_title': persona_title or 'VIP Champions',
                'email_subject': 'Exclusive VIP Invitation: Early Access & Premier Concierge Rewards',
                'email_body': "Hi {{First_Name}},\n\nAs one of our most valued VIP Champions, we want to thank you for your exceptional loyalty. You've unlocked exclusive early access to our upcoming high-end collection before anyone else, plus complimentary express shipping on your next order.\n\nClaim your VIP perk today with code: VIPHERO.",
                'sms_text': "VIP Perk: You have early access to our private collection + free express shipping! Use code VIPHERO.",
                'ad_headline': "Join the Elite VIP Circle — Early Product Drops & Concierge Rewards",
                'ad_description': "Enjoy exclusive early access, dedicated support, and premium rewards crafted for top brand champions.",
                'call_to_action': "Claim VIP Early Access",
                'discount_offer': "Free Express Shipping & 15% VIP Access Credit"
            },
            'AT_RISK': {
                'persona_key': 'AT_RISK',
                'persona_title': persona_title or 'At-Risk / Hibernating',
                'email_subject': 'We miss you, {{First_Name}} — Here is $25 off your next order!',
                'email_body': "Hi {{First_Name}},\n\nNotice anything missing? It's been a while since your last order, and we would love to welcome you back. Enjoy $25 off your next order of $75+ with personalized recommendations just for you.\n\nUse code: WELCOME25 at checkout.",
                'sms_text': "We miss you! Take $25 off your next order with code WELCOME25. Valid for 48 hours.",
                'ad_headline': "Welcome Back Offer: $25 Off Your Next Order",
                'ad_description': "Re-discover your favorite products with an exclusive $25 credit on us.",
                'call_to_action': "Reactivate Your Credit",
                'discount_offer': "$25 Voucher on $75+ Orders"
            },
            'BARGAIN': {
                'persona_key': 'BARGAIN',
                'persona_title': persona_title or 'Bargain Hunters',
                'email_subject': 'Flash Clearance Alert: Up to 50% Off Top-Rated Bundles!',
                'email_body': "Hi {{First_Name}},\n\nGreat news for savvy shoppers! Our major seasonal clearance sale is live now. Get up to 50% off multi-item bundles and save big before stock runs out.\n\nShop the flash sale now with code: BUNDLESAVE.",
                'sms_text': "Flash Sale Alert! Up to 50% off select bundles today only. Code: BUNDLESAVE",
                'ad_headline': "Savvy Savings: Up to 50% Off Multi-Item Bundles",
                'ad_description': "Maximum savings on high-demand essentials. Limited-time bundle deals live now.",
                'call_to_action': "Shop Flash Sale",
                'discount_offer': "Up to 50% Off Bundle Deals"
            },
            'LOYALIST': {
                'persona_key': 'LOYALIST',
                'persona_title': persona_title or 'Steady Loyalists',
                'email_subject': 'Double Loyalty Points Active + Tier Upgrade Preview',
                'email_body': "Hi {{First_Name}},\n\nYour steady support means the world to us. For the next 3 days, earn 2X loyalty points on every purchase and move one step closer to Gold Status.\n\nUnlock double points now with code: DOUBLEPOINTS.",
                'sms_text': "2X Loyalty Points Active! Earn double points on all orders for 3 days. Code: DOUBLEPOINTS",
                'ad_headline': "Double Points Weekend — Level Up Your Loyalty Rewards",
                'ad_description': "Earn 2X rewards on your favorite categories and unlock tier perks faster.",
                'call_to_action': "Earn Double Points",
                'discount_offer': "2X Reward Points Accelerator"
            },
            'NEW_BUYER': {
                'persona_key': 'NEW_BUYER',
                'persona_title': persona_title or 'New Potential Loyalists',
                'email_subject': 'Welcome to the Family! Here is 15% Off Your 2nd Order',
                'email_body': "Hi {{First_Name}},\n\nThank you for choosing us for your recent order! We hope you loved your purchase. To celebrate your second order, here is 15% off your next checkout.\n\nUse code: HELLO15 to claim your welcome gift.",
                'sms_text': "Welcome gift! Enjoy 15% off your 2nd purchase with code HELLO15.",
                'ad_headline': "Welcome Gift: 15% Off Your Second Purchase",
                'ad_description': "Discover top-rated companion items with your newcomer welcome discount.",
                'call_to_action': "Claim Welcome 15% Off",
                'discount_offer': "15% Off 2nd Purchase Welcome Credit"
            }
        }

        return catalog.get(persona_key, catalog['LOYALIST'])
