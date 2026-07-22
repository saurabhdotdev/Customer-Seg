import os
import numpy as np
import pandas as pd

def generate_customer_dataset(n_samples: int = 5000, random_state: int = 42, output_path: str = None) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer segmentation dataset with distinct behavioral profiles
    (Champions, Loyalists, At-Risk, Bargain Hunters, New Buyers) plus natural noise.
    """
    np.random.seed(random_state)
    
    # Archetype proportions
    n_champions = int(n_samples * 0.15)
    n_loyalists = int(n_samples * 0.25)
    n_at_risk = int(n_samples * 0.20)
    n_bargain = int(n_samples * 0.20)
    n_new = int(n_samples * 0.15)
    n_random = n_samples - (n_champions + n_loyalists + n_at_risk + n_bargain + n_new)
    
    profiles = []
    
    # 1. Champions (High Spend, High Frequency, Very Recent)
    for _ in range(n_champions):
        recency = np.random.randint(1, 30)
        frequency = np.random.randint(35, 110)
        monetary = np.random.uniform(3500, 18000)
        category_diversity = np.random.randint(6, 10)
        engagement = np.random.uniform(75, 99)
        support_tickets = np.random.randint(0, 3)
        discount_ratio = np.random.uniform(0.05, 0.25)
        return_rate = np.random.uniform(0.01, 0.08)
        age = np.random.randint(25, 55)
        channel = np.random.choice(["Mobile App", "Web"], p=[0.7, 0.3])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    # 2. Loyalists (Steady Frequency, Moderate-High Spend, Recent)
    for _ in range(n_loyalists):
        recency = np.random.randint(10, 60)
        frequency = np.random.randint(18, 45)
        monetary = np.random.uniform(1500, 5000)
        category_diversity = np.random.randint(4, 8)
        engagement = np.random.uniform(60, 85)
        support_tickets = np.random.randint(0, 4)
        discount_ratio = np.random.uniform(0.15, 0.40)
        return_rate = np.random.uniform(0.02, 0.12)
        age = np.random.randint(22, 62)
        channel = np.random.choice(["Web", "Mobile App", "In-Store"], p=[0.5, 0.4, 0.1])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    # 3. At-Risk / Hibernating (High Past Spend/Freq, High Recency Gap, High Support Tickets)
    for _ in range(n_at_risk):
        recency = np.random.randint(120, 365)
        frequency = np.random.randint(15, 60)
        monetary = np.random.uniform(1800, 9000)
        category_diversity = np.random.randint(3, 7)
        engagement = np.random.uniform(10, 45)
        support_tickets = np.random.randint(3, 12)
        discount_ratio = np.random.uniform(0.20, 0.50)
        return_rate = np.random.uniform(0.08, 0.25)
        age = np.random.randint(30, 68)
        channel = np.random.choice(["Web", "In-Store"], p=[0.6, 0.4])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    # 4. Bargain Hunters (Low-Medium Spend, High Discount Usage, High Frequency during sales)
    for _ in range(n_bargain):
        recency = np.random.randint(15, 150)
        frequency = np.random.randint(12, 50)
        monetary = np.random.uniform(300, 1800)
        category_diversity = np.random.randint(2, 5)
        engagement = np.random.uniform(40, 75)
        support_tickets = np.random.randint(1, 5)
        discount_ratio = np.random.uniform(0.65, 0.98)
        return_rate = np.random.uniform(0.05, 0.18)
        age = np.random.randint(18, 48)
        channel = np.random.choice(["Mobile App", "Web"], p=[0.6, 0.4])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    # 5. New Buyers (Very Recent, Low Frequency 1-3, Moderate Spend)
    for _ in range(n_new):
        recency = np.random.randint(1, 25)
        frequency = np.random.randint(1, 4)
        monetary = np.random.uniform(50, 450)
        category_diversity = np.random.randint(1, 3)
        engagement = np.random.uniform(50, 80)
        support_tickets = np.random.randint(0, 2)
        discount_ratio = np.random.uniform(0.10, 0.35)
        return_rate = np.random.uniform(0.0, 0.05)
        age = np.random.randint(20, 50)
        channel = np.random.choice(["Mobile App", "Web", "In-Store"], p=[0.5, 0.3, 0.2])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    # 6. Random Noise / Outliers
    for _ in range(n_random):
        recency = np.random.randint(1, 365)
        frequency = np.random.randint(1, 80)
        monetary = np.random.uniform(20, 12000)
        category_diversity = np.random.randint(1, 9)
        engagement = np.random.uniform(5, 95)
        support_tickets = np.random.randint(0, 8)
        discount_ratio = np.random.uniform(0.0, 1.0)
        return_rate = np.random.uniform(0.0, 0.30)
        age = np.random.randint(18, 70)
        channel = np.random.choice(["Web", "Mobile App", "In-Store"], p=[0.4, 0.4, 0.2])
        profiles.append((recency, frequency, monetary, category_diversity, engagement, support_tickets, discount_ratio, return_rate, age, channel))

    cols = ["Recency_Days", "Frequency_Orders", "Monetary_Spend", "Category_Diversity", 
            "Engagement_Score", "Support_Tickets", "Discount_Ratio", "Return_Rate", "Age", "Preferred_Channel"]
    
    df = pd.DataFrame(profiles, columns=cols)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Customer IDs
    df.insert(0, "Customer_ID", [f"CUST-{10001 + i}" for i in range(len(df))])
    
    # Derived Feature: Average Order Value (AOV)
    df["Avg_Order_Value"] = (df["Monetary_Spend"] / df["Frequency_Orders"]).round(2)
    
    # Gender assignment
    genders = ["Female", "Male", "Non-Binary"]
    df["Gender"] = np.random.choice(genders, size=len(df), p=[0.51, 0.46, 0.03])
    
    # Churn Risk Calculation (synthetic domain logic)
    df["Churn_Risk_Index"] = (
        0.50 * (df["Recency_Days"] / 365.0) +
        0.25 * (1.0 - (df["Engagement_Score"] / 100.0)) +
        0.15 * (df["Support_Tickets"] / 12.0) +
        0.10 * (df["Return_Rate"] / 0.30)
    ).clip(0.0, 1.0).round(4)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} customer records saved to {output_path}")

    return df

if __name__ == "__main__":
    generate_customer_dataset(output_path="../../data/raw/customer_transactions.csv")
