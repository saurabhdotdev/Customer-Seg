from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerInputSchema(BaseModel):
    Recency_Days: int = Field(..., ge=0, le=3650, description="Days since last purchase")
    Frequency_Orders: int = Field(..., ge=0, le=5000, description="Total order count in last 12 months")
    Monetary_Spend: float = Field(..., ge=0.0, description="Total spend in USD")
    Category_Diversity: int = Field(default=3, ge=1, le=50, description="Number of distinct categories")
    Engagement_Score: float = Field(default=70.0, ge=0.0, le=100.0, description="Engagement score (1-100)")
    Support_Tickets: int = Field(default=1, ge=0, le=500, description="Support tickets submitted")
    Discount_Ratio: float = Field(default=0.20, ge=0.0, le=1.0, description="Ratio of discounted purchases")
    Return_Rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Product return rate")
    Age: int = Field(default=35, ge=1, le=120)
    Preferred_Channel: str = Field(default="Mobile App", description="Preferred channel: Web, Mobile App, In-Store")
    Gender: str = Field(default="Female", description="Gender: Female, Male, Non-Binary")
    Customer_LTV: Optional[float] = Field(default=5000.0, description="Customer lifetime value")
    Region: Optional[str] = Field(default="North America", description="Customer geographical region")
    App_Sessions_Per_Month: Optional[int] = Field(default=20, description="Monthly app sessions")
    NPS_Score: Optional[int] = Field(default=8, description="Net promoter score (1-10)")

class PredictionResponseSchema(BaseModel):
    predicted_cluster: int
    confidence_score: float
    predicted_ltv_12m: float
    persona_key: str
    persona_title: str
    tagline: str
    description: str
    recommended_strategy: str
    color: str
    icon: str
    churn_risk_index: float
    is_anomaly: bool
    anomaly_score: Optional[float] = 0.0
    anomaly_type: Optional[str] = "Normal Pattern"
    churn_explainability: Optional[Dict[str, Any]] = None
    pca_coordinates: List[float]
    metrics_summary: Dict[str, Any]

class CampaignCopyRequestSchema(BaseModel):
    persona_key: str = Field(..., description="Persona key: CHAMPION, LOYALIST, AT_RISK, BARGAIN, NEW_BUYER")
    persona_title: Optional[str] = Field(default="", description="Persona display title")

class CampaignCopyResponseSchema(BaseModel):
    persona_key: str
    persona_title: str
    email_subject: str
    email_body: str
    sms_text: str
    ad_headline: str
    ad_description: str
    call_to_action: str
    discount_offer: str


class NextBestActionRequestSchema(BaseModel):
    customer: CustomerInputSchema
    churn_risk_score: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)
    predicted_ltv_12m: Optional[float] = Field(default=1200.0, ge=0.0)
    is_anomaly: Optional[bool] = False
    persona_key: Optional[str] = "CHAMPION"


class NextBestActionResponseSchema(BaseModel):
    total_actions_evaluated: int
    top_action: Optional[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    evaluated_metrics: Dict[str, Any]

