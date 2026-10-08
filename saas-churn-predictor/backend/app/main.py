import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AccountFeatures, PredictionResponse, DriverContribution, ModelInfo

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CATEGORICAL = ["plan_tier", "contract_type"]
NUMERIC = [
    "seats", "tenure_months", "monthly_price", "monthly_logins_per_seat",
    "feature_adoption_pct", "days_since_last_login", "support_tickets_last_90d",
    "avg_ticket_resolution_hours", "nps_score", "had_payment_failure_last_90d",
    "discount_pct", "onboarding_completed", "integrations_connected",
    "champion_left_company",
]

FEATURE_LABELS = {
    "days_since_last_login": "Days since last login",
    "feature_adoption_pct": "Feature adoption",
    "tenure_months": "Tenure",
    "nps_score": "NPS score",
    "support_tickets_last_90d": "Support tickets (90d)",
    "had_payment_failure_last_90d": "Recent payment failure",
    "champion_left_company": "Champion left the company",
    "onboarding_completed": "Onboarding completed",
    "integrations_connected": "Integrations connected",
    "contract_type_Monthly": "Monthly contract",
    "contract_type_Annual": "Annual contract",
    "plan_tier_Enterprise": "Enterprise plan",
    "plan_tier_Growth": "Growth plan",
    "plan_tier_Starter": "Starter plan",
    "avg_ticket_resolution_hours": "Avg. ticket resolution time",
    "monthly_logins_per_seat": "Logins per seat / month",
    "seats": "Seats",
    "monthly_price": "Monthly price",
    "discount_pct": "Discount applied",
}

app = FastAPI(title="SaaS Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load artifacts once at startup ----
pipe = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

background_df = pd.read_csv(os.path.join(MODEL_DIR, "saas_churn_dataset.csv")).sample(
    n=200, random_state=42
)
background_transformed = pipe.named_steps["prep"].transform(
    background_df[CATEGORICAL + NUMERIC]
)

full_df = pd.read_csv(os.path.join(MODEL_DIR, "saas_churn_dataset.csv"))

clf = pipe.named_steps["clf"]
model_type = type(clf).__name__
if model_type == "LogisticRegression":
    explainer = shap.LinearExplainer(clf, background_transformed)
elif model_type in ("RandomForestClassifier",):
    explainer = shap.TreeExplainer(clf)
else:
    explainer = shap.Explainer(clf, background_transformed)


def _risk_tier(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    if prob < 0.35:
        return "Medium"
    if prob < 0.6:
        return "High"
    return "Critical"


def _format_value(feature: str, row: dict) -> str:
    base = feature.split("_Starter")[0].split("_Growth")[0].split("_Enterprise")[0]
    base = feature.split("_Monthly")[0].split("_Annual")[0] if "contract_type" in feature else base
    if feature in row:
        v = row[feature]
        if isinstance(v, bool) or feature in ("had_payment_failure_last_90d", "onboarding_completed", "champion_left_company"):
            return "Yes" if v else "No"
        return str(v)
    return "Yes"


def _compute_portfolio_stats():
    X_full = full_df[CATEGORICAL + NUMERIC].copy()
    probs = pipe.predict_proba(X_full)[:, 1]

    churn_by_plan = (
        full_df.assign(pred_prob=probs)
        .groupby("plan_tier")["pred_prob"].mean()
        .round(4).to_dict()
    )
    churn_by_contract = (
        full_df.assign(pred_prob=probs)
        .groupby("contract_type")["pred_prob"].mean()
        .round(4).to_dict()
    )

    tiers = pd.cut(
        probs, bins=[-0.01, 0.15, 0.35, 0.6, 1.01],
        labels=["Low", "Medium", "High", "Critical"]
    )
    risk_distribution = tiers.value_counts().reindex(["Low", "Medium", "High", "Critical"]).to_dict()

    # Global feature importance: |coefficient| for linear models on standardized inputs,
    # or built-in feature_importances_ for tree models.
    if hasattr(clf, "coef_"):
        raw_importance = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        raw_importance = clf.feature_importances_
    else:
        raw_importance = np.zeros(len(feature_names))

    global_drivers = sorted(
        zip(feature_names, raw_importance), key=lambda x: x[1], reverse=True
    )[:8]
    global_drivers = [
        {"feature": f, "label": FEATURE_LABELS.get(f, f.replace("_", " ").capitalize()), "importance": round(float(v), 4)}
        for f, v in global_drivers
    ]

    churned_mask = full_df["churned"] == 1
    cohort_compare = {
        "tenure_months": {
            "churned": round(float(full_df.loc[churned_mask, "tenure_months"].mean()), 1),
            "retained": round(float(full_df.loc[~churned_mask, "tenure_months"].mean()), 1),
        },
        "nps_score": {
            "churned": round(float(full_df.loc[churned_mask, "nps_score"].mean()), 1),
            "retained": round(float(full_df.loc[~churned_mask, "nps_score"].mean()), 1),
        },
        "days_since_last_login": {
            "churned": round(float(full_df.loc[churned_mask, "days_since_last_login"].mean()), 1),
            "retained": round(float(full_df.loc[~churned_mask, "days_since_last_login"].mean()), 1),
        },
        "feature_adoption_pct": {
            "churned": round(float(full_df.loc[churned_mask, "feature_adoption_pct"].mean()), 1),
            "retained": round(float(full_df.loc[~churned_mask, "feature_adoption_pct"].mean()), 1),
        },
    }

    return {
        "churn_by_plan": churn_by_plan,
        "churn_by_contract": churn_by_contract,
        "risk_distribution": {k: int(v) for k, v in risk_distribution.items()},
        "global_drivers": global_drivers,
        "cohort_compare": cohort_compare,
        "n_accounts": len(full_df),
    }


PORTFOLIO_STATS = _compute_portfolio_stats()



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return metrics


@app.get("/portfolio-stats")
def portfolio_stats():
    return PORTFOLIO_STATS


@app.post("/predict", response_model=PredictionResponse)
def predict(account: AccountFeatures):
    try:
        row = account.model_dump()
        X = pd.DataFrame([row])[CATEGORICAL + NUMERIC]
        # cast bools to int for the model, matching training encoding
        for col in ["had_payment_failure_last_90d", "onboarding_completed", "champion_left_company"]:
            X[col] = X[col].astype(int)

        proba = float(pipe.predict_proba(X)[0, 1])

        Xt = pipe.named_steps["prep"].transform(X)
        sv = explainer(Xt)
        values = np.array(sv.values).reshape(-1)

        contributions = list(zip(feature_names, values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_drivers = []
        for feat, impact in contributions[:5]:
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").capitalize())
            top_drivers.append(
                DriverContribution(
                    feature=feat,
                    label=label,
                    value=_format_value(feat, row),
                    impact=round(float(impact), 4),
                )
            )

        return PredictionResponse(
            churn_probability=round(proba, 4),
            risk_tier=_risk_tier(proba),
            top_drivers=top_drivers,
            model_name=metrics["best_model"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
