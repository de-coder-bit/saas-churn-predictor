"""
Trains a churn classifier on the synthetic SaaS dataset and exports:
  - churn_model.joblib        (full sklearn Pipeline: preprocessing + model)
  - feature_names.joblib      (post-encoding feature names, for SHAP display)
  - metrics.json              (evaluation metrics for the README / API /model-info)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

DATA_PATH = "/home/claude/saas-churn-predictor/backend/app/model/saas_churn_dataset.csv"

CATEGORICAL = ["plan_tier", "contract_type"]
NUMERIC = [
    "seats", "tenure_months", "monthly_price", "monthly_logins_per_seat",
    "feature_adoption_pct", "days_since_last_login", "support_tickets_last_90d",
    "avg_ticket_resolution_hours", "nps_score", "had_payment_failure_last_90d",
    "discount_pct", "onboarding_completed", "integrations_connected",
    "champion_left_company",
]
TARGET = "churned"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42,
        ),
    }

    results = {}
    best_name, best_pipeline, best_auc = None, None, -1

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            "roc_auc": round(auc, 4),
            "f1": round(f1_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
        }
        print(f"\n== {name} ==")
        print(results[name])
        print(classification_report(y_test, preds))
        if auc > best_auc:
            best_auc, best_name, best_pipeline = auc, name, pipe

    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")

    # ---- Calibration correction ----
    # class_weight='balanced' / scale_pos_weight effectively fits the model as if
    # classes were 50/50, which inflates predicted probabilities relative to the
    # true churn rate. For logistic regression this is a simple, well-known fix:
    # shift the intercept using the case-control correction (Prentice & Pyke 1979)
    # so predict_proba reflects the true population prior. For tree-based models
    # we recalibrate with isotonic regression via CalibratedClassifierCV instead.
    true_churn_rate = float(y.mean())

    if best_name == "logistic_regression":
        final_pipe = Pipeline([("prep", build_preprocessor()), ("clf", candidates[best_name])])
        final_pipe.fit(X, y)
        logit_correction = np.log(true_churn_rate / (1 - true_churn_rate))  # sample prior was 50/50 -> ln(1)=0 baseline
        final_pipe.named_steps["clf"].intercept_ = final_pipe.named_steps["clf"].intercept_ + logit_correction

        # recompute calibrated metrics on the held-out test set for accurate reporting
        test_pipe = Pipeline([("prep", build_preprocessor()), ("clf", candidates[best_name].__class__(**candidates[best_name].get_params()))])
        test_pipe.fit(X_train, y_train)
        test_pipe.named_steps["clf"].intercept_ = test_pipe.named_steps["clf"].intercept_ + logit_correction
        calibrated_proba = test_pipe.predict_proba(X_test)[:, 1]
        calibrated_preds = (calibrated_proba >= 0.5).astype(int)
        results[best_name] = {
            "roc_auc": round(roc_auc_score(y_test, calibrated_proba), 4),  # AUC unaffected by monotonic shift
            "f1": round(f1_score(y_test, calibrated_preds), 4),
            "precision": round(precision_score(y_test, calibrated_preds), 4),
            "recall": round(recall_score(y_test, calibrated_preds), 4),
        }
        print(f"Calibrated metrics @0.5 threshold: {results[best_name]}")
    else:
        base_pipe = Pipeline([("prep", build_preprocessor()), ("clf", candidates[best_name])])
        final_pipe = CalibratedClassifierCV(base_pipe, method="isotonic", cv=5)
        final_pipe.fit(X, y)

    joblib.dump(final_pipe, "/home/claude/saas-churn-predictor/backend/app/model/churn_model.joblib")

    fitted_preprocessor = build_preprocessor().fit(X)
    feature_names = (
        NUMERIC
        + list(fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    )
    joblib.dump(feature_names, "/home/claude/saas-churn-predictor/backend/app/model/feature_names.joblib")

    with open("/home/claude/saas-churn-predictor/backend/app/model/metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "churn_rate": round(float(y.mean()), 4),
            "n_samples": len(df),
        }, f, indent=2)

    print("Artifacts saved.")


if __name__ == "__main__":
    main()
