from pydantic import BaseModel, Field
from typing import Literal, List


class AccountFeatures(BaseModel):
    plan_tier: Literal["Starter", "Growth", "Enterprise"]
    contract_type: Literal["Monthly", "Annual"]
    seats: int = Field(ge=1, le=2000)
    tenure_months: int = Field(ge=0, le=200)
    monthly_price: float = Field(ge=0, le=100000)
    monthly_logins_per_seat: float = Field(ge=0, le=200)
    feature_adoption_pct: float = Field(ge=0, le=100)
    days_since_last_login: int = Field(ge=0, le=365)
    support_tickets_last_90d: int = Field(ge=0, le=100)
    avg_ticket_resolution_hours: float = Field(ge=0, le=2000)
    nps_score: int = Field(ge=-100, le=100)
    had_payment_failure_last_90d: bool
    discount_pct: int = Field(ge=0, le=100)
    onboarding_completed: bool
    integrations_connected: int = Field(ge=0, le=100)
    champion_left_company: bool

    class Config:
        json_schema_extra = {
            "example": {
                "plan_tier": "Growth",
                "contract_type": "Monthly",
                "seats": 12,
                "tenure_months": 8,
                "monthly_price": 149.0,
                "monthly_logins_per_seat": 6.5,
                "feature_adoption_pct": 34.0,
                "days_since_last_login": 21,
                "support_tickets_last_90d": 4,
                "avg_ticket_resolution_hours": 40.0,
                "nps_score": -10,
                "had_payment_failure_last_90d": True,
                "discount_pct": 0,
                "onboarding_completed": False,
                "integrations_connected": 1,
                "champion_left_company": False,
            }
        }


class DriverContribution(BaseModel):
    feature: str
    label: str
    value: str
    impact: float  # signed SHAP contribution (positive = pushes toward churn)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    top_drivers: List[DriverContribution]
    model_name: str


class ModelInfo(BaseModel):
    best_model: str
    all_results: dict
    churn_rate: float
    n_samples: int
