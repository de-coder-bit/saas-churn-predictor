"""
Generates a synthetic SaaS B2B customer churn dataset.

Simulates a subscription SaaS business (like a project-management or
CRM tool) with account-level features that genuinely drive churn:
plan tier, seat count, usage, support friction, NPS, contract type,
tenure, and billing behavior. The churn label is generated from a
logistic function of these features plus noise, so the signal is
realistic but not trivially separable.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 6000


def generate_dataset(n=N):
    plan_tier = np.random.choice(
        ["Starter", "Growth", "Enterprise"], size=n, p=[0.45, 0.4, 0.15]
    )
    plan_base_price = {"Starter": 29, "Growth": 99, "Enterprise": 399}

    contract_type = np.random.choice(
        ["Monthly", "Annual"], size=n, p=[0.6, 0.4]
    )

    seats = np.clip(
        np.round(
            np.where(
                plan_tier == "Starter",
                np.random.gamma(2, 1.5, n),
                np.where(
                    plan_tier == "Growth",
                    np.random.gamma(4, 2.5, n),
                    np.random.gamma(8, 4, n),
                )
            )
        ),
        1, 500
    ).astype(int)

    tenure_months = np.clip(np.random.exponential(18, n), 1, 96).round().astype(int)

    # Usage: logins per user per month - healthy accounts use the product more
    monthly_logins_per_seat = np.clip(np.random.normal(12, 6, n), 0, 60)

    # Feature adoption score 0-100: % of core features actually used
    feature_adoption_pct = np.clip(np.random.normal(55, 22, n), 2, 100)

    days_since_last_login = np.clip(np.random.exponential(6, n), 0, 120).round().astype(int)

    support_tickets_last_90d = np.random.poisson(1.4, n)
    # ticket resolution satisfaction (NaN if no tickets)
    avg_ticket_resolution_hours = np.clip(np.random.exponential(20, n), 1, 240)

    nps_score = np.clip(np.random.normal(30, 35, n), -100, 100).round().astype(int)

    had_payment_failure_last_90d = np.random.binomial(1, 0.12, n)

    discount_pct = np.random.choice(
        [0, 10, 20, 30], size=n, p=[0.55, 0.2, 0.15, 0.1]
    )

    onboarding_completed = np.random.binomial(1, 0.72, n)

    integrations_connected = np.clip(np.random.poisson(2.1, n), 0, 12)

    monthly_price = np.array([plan_base_price[p] for p in plan_tier]) * (
        1 - discount_pct / 100
    ) + seats * np.where(plan_tier == "Enterprise", 8, np.where(plan_tier == "Growth", 4, 1.5))

    champion_left_company = np.random.binomial(1, 0.08, n)

    # ---- Churn probability: logistic combination of real drivers ----
    z = (
        -1.6
        + 0.11 * days_since_last_login
        - 0.16 * feature_adoption_pct / 10
        - 0.85 * (tenure_months / 12)
        - 0.09 * nps_score / 10
        + 0.95 * support_tickets_last_90d / 3
        + 1.5 * had_payment_failure_last_90d
        + 1.3 * champion_left_company
        - 1.1 * onboarding_completed
        - 0.7 * integrations_connected / 3
        + 0.75 * (contract_type == "Monthly").astype(int)
        - 0.55 * (plan_tier == "Enterprise").astype(int)
        + 0.03 * avg_ticket_resolution_hours / 10
        - 0.32 * monthly_logins_per_seat / 10
        + np.random.normal(0, 0.45, n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churned = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "account_id": [f"ACC-{10000+i}" for i in range(n)],
        "plan_tier": plan_tier,
        "contract_type": contract_type,
        "seats": seats,
        "tenure_months": tenure_months,
        "monthly_price": monthly_price.round(2),
        "monthly_logins_per_seat": monthly_logins_per_seat.round(1),
        "feature_adoption_pct": feature_adoption_pct.round(1),
        "days_since_last_login": days_since_last_login,
        "support_tickets_last_90d": support_tickets_last_90d,
        "avg_ticket_resolution_hours": avg_ticket_resolution_hours.round(1),
        "nps_score": nps_score,
        "had_payment_failure_last_90d": had_payment_failure_last_90d,
        "discount_pct": discount_pct,
        "onboarding_completed": onboarding_completed,
        "integrations_connected": integrations_connected,
        "champion_left_company": champion_left_company,
        "churned": churned,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/saas-churn-predictor/backend/app/model/saas_churn_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(df.head())
