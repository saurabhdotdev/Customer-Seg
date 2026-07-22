from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerInputSchema(BaseModel):
    Recency_Days: int = Field(..., ge=1, le=365, description="Days since last purchase")
    Frequency_Orders: int = Field(..., ge=1, le=200, description="Total order count in last 12 months")
    Monetary_Spend: float = Field(..., ge=1.0, description="Total spend in USD")
    Category_Diversity: int = Field(default=3, ge=1, le=10, description="Number of distinct categories")
    Engagement_Score: float = Field(default=70.0, ge=1.0, le=100.0, description="Engagement score (1-100)")
    Support_Tickets: int = Field(default=1, ge=0, le=20, description="Support tickets submitted")
    Discount_Ratio: float = Field(default=0.20, ge=0.0, le=1.0, description="Ratio of discounted purchases")
    Return_Rate: float = Field(default=0.05, ge=0.0, le=0.50, description="Product return rate")
    Age: int = Field(default=35, ge=18, le=90)
    Preferred_Channel: str = Field(default="Mobile App", description="Preferred channel: Web, Mobile App, In-Store")
    Gender: str = Field(default="Female", description="Gender: Female, Male, Non-Binary")

class PredictionResponseSchema(BaseModel):
    predicted_cluster: int
    persona_key: str
    persona_title: str
    tagline: str
    description: str
    recommended_strategy: str
    color: str
    icon: str
    churn_risk_index: float
    pca_coordinates: List[float]
    metrics_summary: Dict[str, Any]
