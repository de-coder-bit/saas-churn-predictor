import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# ---- Palette (mirrors the instrument-panel design) ----
INK = "#10262A"
PANEL = "#17383D"
HAIRLINE = "#2E5257"
PAPER = "#EDEAE1"
PAPER_DIM = "#A9BFBE"
AMBER = "#E8A33D"
TEAL = "#6FBFB0"
CORAL = "#D9614F"
MEDIUM = "#C9C15A"

PRESETS = {
    "At-risk account": {
        "plan_tier": "Growth", "contract_type": "Monthly", "seats": 12, "tenure_months": 8,
        "monthly_price": 149.0, "monthly_logins_per_seat": 6.5, "feature_adoption_pct": 34,
        "days_since_last_login": 21, "support_tickets_last_90d": 4, "avg_ticket_resolution_hours": 40.0,
        "nps_score": -10, "had_payment_failure_last_90d": True, "discount_pct": 0,
        "onboarding_completed": False, "integrations_connected": 1, "champion_left_company": False,
    },
    "Healthy account": {
        "plan_tier": "Enterprise", "contract_type": "Annual", "seats": 80, "tenure_months": 30,
        "monthly_price": 990.0, "monthly_logins_per_seat": 22.0, "feature_adoption_pct": 78,
        "days_since_last_login": 1, "support_tickets_last_90d": 0, "avg_ticket_resolution_hours": 6.0,
        "nps_score": 62, "had_payment_failure_last_90d": False, "discount_pct": 10,
        "onboarding_completed": True, "integrations_connected": 6, "champion_left_company": False,
    },
    "Critical account": {
        "plan_tier": "Starter", "contract_type": "Monthly", "seats": 3, "tenure_months": 2,
        "monthly_price": 29.0, "monthly_logins_per_seat": 1.2, "feature_adoption_pct": 9,
        "days_since_last_login": 45, "support_tickets_last_90d": 6, "avg_ticket_resolution_hours": 96.0,
        "nps_score": -60, "had_payment_failure_last_90d": True, "discount_pct": 0,
        "onboarding_completed": False, "integrations_connected": 0, "champion_left_company": True,
    },
}


@st.cache_resource
def load_artifacts():
    # Train the model at startup from the CSV (takes ~1-2 seconds).
    # This avoids scikit-learn version mismatches with a pickled model file.
    full_df = pd.read_csv(os.path.join(MODEL_DIR, "saas_churn_dataset.csv"))
    X = full_df[CATEGORICAL + NUMERIC]
    y = full_df["churned"]

    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X, y)

    # Calibration fix: balanced class weights inflate probabilities,
    # so shift the intercept back to the true churn rate.
    rate = float(y.mean())
    pipe.named_steps["clf"].intercept_ = pipe.named_steps["clf"].intercept_ + np.log(rate / (1 - rate))

    feature_names = NUMERIC + list(
        pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    )

    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    background_df = full_df.sample(n=200, random_state=42)
    background_transformed = pipe.named_steps["prep"].transform(background_df[CATEGORICAL + NUMERIC])

    clf = pipe.named_steps["clf"]
    explainer = shap.LinearExplainer(clf, background_transformed)

    return pipe, feature_names, metrics, full_df, explainer, clf


@st.cache_data
def compute_portfolio_stats(_pipe, _clf, feature_names, full_df):
    X_full = full_df[CATEGORICAL + NUMERIC].copy()
    probs = _pipe.predict_proba(X_full)[:, 1]

    churn_by_plan = full_df.assign(p=probs).groupby("plan_tier")["p"].mean().round(4)
    churn_by_contract = full_df.assign(p=probs).groupby("contract_type")["p"].mean().round(4)

    tiers = pd.cut(probs, bins=[-0.01, 0.15, 0.35, 0.6, 1.01], labels=["Low", "Medium", "High", "Critical"])
    risk_distribution = tiers.value_counts().reindex(["Low", "Medium", "High", "Critical"])

    raw_importance = np.abs(_clf.coef_[0]) if hasattr(_clf, "coef_") else _clf.feature_importances_
    global_drivers = sorted(zip(feature_names, raw_importance), key=lambda x: x[1], reverse=True)[:8]

    churned_mask = full_df["churned"] == 1
    cohort_cols = ["tenure_months", "nps_score", "days_since_last_login", "feature_adoption_pct"]
    cohort_compare = pd.DataFrame({
        "Churned": full_df.loc[churned_mask, cohort_cols].mean().round(1),
        "Retained": full_df.loc[~churned_mask, cohort_cols].mean().round(1),
    })

    return churn_by_plan, churn_by_contract, risk_distribution, global_drivers, cohort_compare


def risk_tier(prob):
    if prob < 0.15:
        return "Low", TEAL
    if prob < 0.35:
        return "Medium", MEDIUM
    if prob < 0.6:
        return "High", AMBER
    return "Critical", CORAL


def make_gauge(prob):
    tier, color = risk_tier(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": PAPER, "size": 44, "family": "monospace"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PAPER_DIM, "tickfont": {"color": PAPER_DIM, "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15], "color": "rgba(111,191,176,0.15)"},
                {"range": [15, 35], "color": "rgba(201,193,90,0.15)"},
                {"range": [35, 60], "color": "rgba(232,163,61,0.15)"},
                {"range": [60, 100], "color": "rgba(217,97,79,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": PAPER, "family": "monospace"},
    )
    return fig, tier, color


def driver_chart(top_drivers):
    labels = [d[0] for d in top_drivers][::-1]
    values = [d[1] for d in top_drivers][::-1]
    colors = [CORAL if v > 0 else TEAL for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in values], textposition="outside",
        textfont={"color": PAPER_DIM, "family": "monospace", "size": 11},
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor=HAIRLINE, zeroline=True, zerolinecolor=HAIRLINE,
                   tickfont={"color": PAPER_DIM, "size": 10}),
        yaxis=dict(tickfont={"color": PAPER, "size": 12}),
        font={"family": "sans-serif"},
    )
    return fig


def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: {INK}; }}
    .block-container {{ padding-top: 2rem; max-width: 1100px; }}
    h1, h2, h3 {{ font-family: Georgia, serif; color: {PAPER}; }}
    .stMarkdown, p, label, .stCaption {{ color: {PAPER_DIM}; }}
    .brand-tag {{ font-family: monospace; letter-spacing: 3px; color: {AMBER}; font-size: 13px; }}
    div[data-testid="stMetricValue"] {{ color: {PAPER}; font-family: monospace; }}
    div[data-testid="stMetricLabel"] {{ color: {PAPER_DIM}; }}
    .risk-badge {{
        display: inline-block; font-family: monospace; letter-spacing: 1.5px;
        text-transform: uppercase; padding: 6px 16px; border-radius: 999px;
        border: 1px solid; text-align: center; font-size: 12px; margin-top: -10px;
    }}
    hr {{ border-color: {HAIRLINE}; }}
    </style>
    """, unsafe_allow_html=True)


def render_predict_tab(pipe, feature_names, metrics, explainer):
    if "form" not in st.session_state:
        st.session_state.form = dict(PRESETS["At-risk account"])

    st.markdown('<p class="brand-tag">◎ ACCOUNT MANIFEST</p>', unsafe_allow_html=True)
    preset_cols = st.columns(3)
    for i, name in enumerate(PRESETS):
        if preset_cols[i].button(name, use_container_width=True):
            st.session_state.form = dict(PRESETS[name])

    f = st.session_state.form

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        c1, c2 = st.columns(2)
        f["plan_tier"] = c1.selectbox("Plan tier", ["Starter", "Growth", "Enterprise"],
                                       index=["Starter", "Growth", "Enterprise"].index(f["plan_tier"]))
        f["contract_type"] = c2.selectbox("Contract type", ["Monthly", "Annual"],
                                           index=["Monthly", "Annual"].index(f["contract_type"]))
        f["seats"] = c1.number_input("Seats", min_value=1, value=int(f["seats"]))
        f["tenure_months"] = c2.number_input("Tenure (months)", min_value=0, value=int(f["tenure_months"]))
        f["monthly_price"] = c1.number_input("Monthly price ($)", min_value=0.0, value=float(f["monthly_price"]))
        f["discount_pct"] = c2.number_input("Discount applied (%)", min_value=0, max_value=100, value=int(f["discount_pct"]))
        f["monthly_logins_per_seat"] = c1.number_input("Logins per seat / month", min_value=0.0, value=float(f["monthly_logins_per_seat"]))
        f["days_since_last_login"] = c2.number_input("Days since last login", min_value=0, value=int(f["days_since_last_login"]))
        f["feature_adoption_pct"] = st.slider("Feature adoption (%)", 0, 100, int(f["feature_adoption_pct"]))
        f["integrations_connected"] = c1.number_input("Integrations connected", min_value=0, value=int(f["integrations_connected"]))
        f["support_tickets_last_90d"] = c2.number_input("Support tickets (90d)", min_value=0, value=int(f["support_tickets_last_90d"]))
        f["avg_ticket_resolution_hours"] = c1.number_input("Avg. ticket resolution (hrs)", min_value=0.0, value=float(f["avg_ticket_resolution_hours"]))
        f["nps_score"] = st.slider("NPS score", -100, 100, int(f["nps_score"]))
        f["had_payment_failure_last_90d"] = st.checkbox("Payment failure in last 90 days", value=f["had_payment_failure_last_90d"])
        f["onboarding_completed"] = st.checkbox("Onboarding completed", value=f["onboarding_completed"])
        f["champion_left_company"] = st.checkbox("Internal champion left the company", value=f["champion_left_company"])

        run = st.button("Run prediction", type="primary", use_container_width=True)

    with col_result:
        if run or "last_result" in st.session_state:
            row = dict(f)
            X = pd.DataFrame([row])[CATEGORICAL + NUMERIC]
            for col in ["had_payment_failure_last_90d", "onboarding_completed", "champion_left_company"]:
                X[col] = X[col].astype(int)

            proba = float(pipe.predict_proba(X)[0, 1])
            Xt = pipe.named_steps["prep"].transform(X)
            sv = explainer(Xt)
            values = np.array(sv.values).reshape(-1)
            contributions = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)[:5]

            st.session_state.last_result = (proba, contributions)

            fig, tier, color = make_gauge(proba)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="text-align:center;"><span class="risk-badge" style="color:{color};border-color:{color};">{tier} risk</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("&nbsp;")
            st.markdown("**What's driving this** — signed SHAP contribution per signal")
            labeled = [(FEATURE_LABELS.get(feat, feat), val) for feat, val in contributions]
            st.plotly_chart(driver_chart(labeled), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Fill in the manifest and click **Run prediction**, or pick a preset above.")


def render_portfolio_tab(pipe, feature_names, full_df, clf):
    churn_by_plan, churn_by_contract, risk_distribution, global_drivers, cohort_compare = compute_portfolio_stats(
        pipe, clf, feature_names, full_df
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Churn rate by plan tier**")
        fig = go.Figure(go.Bar(x=churn_by_plan.index, y=churn_by_plan.values * 100, marker_color=AMBER))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(title="%", gridcolor=HAIRLINE, tickfont={"color": PAPER_DIM}),
                           xaxis=dict(tickfont={"color": PAPER_DIM}), font={"color": PAPER_DIM})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("**Churn rate by contract type**")
        fig = go.Figure(go.Bar(x=churn_by_contract.index, y=churn_by_contract.values * 100, marker_color=TEAL))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(title="%", gridcolor=HAIRLINE, tickfont={"color": PAPER_DIM}),
                           xaxis=dict(tickfont={"color": PAPER_DIM}), font={"color": PAPER_DIM})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**Portfolio risk distribution** — {len(full_df):,} accounts")
        colors = [TEAL, MEDIUM, AMBER, CORAL]
        fig = go.Figure(go.Bar(
            x=risk_distribution.values, y=risk_distribution.index, orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor=HAIRLINE, tickfont={"color": PAPER_DIM}),
                           yaxis=dict(tickfont={"color": PAPER_DIM}), font={"color": PAPER_DIM})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        st.markdown("**Global model drivers**")
        labels = [FEATURE_LABELS.get(f, f) for f, v in global_drivers][::-1]
        vals = [v for f, v in global_drivers][::-1]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=AMBER))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor=HAIRLINE, tickfont={"color": PAPER_DIM}),
                           yaxis=dict(tickfont={"color": PAPER_DIM, "size": 10}), font={"color": PAPER_DIM})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("**Churned vs. retained: average signals**")
    st.dataframe(cohort_compare, use_container_width=True)


def main():
    st.set_page_config(page_title="Gauge — SaaS Churn", page_icon="◎", layout="wide")
    inject_css()

    pipe, feature_names, metrics, full_df, explainer, clf = load_artifacts()

    st.markdown("### ◎ GAUGE")
    st.markdown('<p class="brand-tag">SAAS CHURN EARLY-WARNING CONSOLE</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio churn rate", f"{metrics['churn_rate']*100:.1f}%")
    m2.metric("Model ROC-AUC", f"{metrics['all_results'][metrics['best_model']]['roc_auc']*100:.1f}")
    m3.metric("Active model", metrics["best_model"].replace("_", " "))

    st.markdown("---")

    tab1, tab2 = st.tabs(["Predict", "Portfolio"])
    with tab1:
        render_predict_tab(pipe, feature_names, metrics, explainer)
    with tab2:
        render_portfolio_tab(pipe, feature_names, full_df, clf)

    st.markdown("---")
    st.caption("Gauge — a synthetic-data demo project. Not connected to a real customer database.")


if __name__ == "__main__":
    main()