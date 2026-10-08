import { useEffect, useState } from "react";
import Gauge from "./Gauge";
import Dashboard from "./Dashboard";
import { predictChurn, getModelInfo } from "./api";
import "./App.css";

const DEFAULT_FORM = {
  plan_tier: "Growth",
  contract_type: "Monthly",
  seats: 12,
  tenure_months: 8,
  monthly_price: 149,
  monthly_logins_per_seat: 6.5,
  feature_adoption_pct: 34,
  days_since_last_login: 21,
  support_tickets_last_90d: 4,
  avg_ticket_resolution_hours: 40,
  nps_score: -10,
  had_payment_failure_last_90d: true,
  discount_pct: 0,
  onboarding_completed: false,
  integrations_connected: 1,
  champion_left_company: false,
};

const PRESETS = {
  "Healthy account": {
    plan_tier: "Enterprise", contract_type: "Annual", seats: 80, tenure_months: 30,
    monthly_price: 990, monthly_logins_per_seat: 22, feature_adoption_pct: 78,
    days_since_last_login: 1, support_tickets_last_90d: 0, avg_ticket_resolution_hours: 6,
    nps_score: 62, had_payment_failure_last_90d: false, discount_pct: 10,
    onboarding_completed: true, integrations_connected: 6, champion_left_company: false,
  },
  "At-risk account": DEFAULT_FORM,
  "Critical account": {
    plan_tier: "Starter", contract_type: "Monthly", seats: 3, tenure_months: 2,
    monthly_price: 29, monthly_logins_per_seat: 1.2, feature_adoption_pct: 9,
    days_since_last_login: 45, support_tickets_last_90d: 6, avg_ticket_resolution_hours: 96,
    nps_score: -60, had_payment_failure_last_90d: true, discount_pct: 0,
    onboarding_completed: false, integrations_connected: 0, champion_left_company: true,
  },
};

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [tab, setTab] = useState("predict");

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const runPrediction = async (payload = form) => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictChurn(payload);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runPrediction(DEFAULT_FORM);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyPreset = (name) => {
    const preset = PRESETS[name];
    setForm(preset);
    runPrediction(preset);
  };

  const onSubmit = (e) => {
    e.preventDefault();
    runPrediction(form);
  };

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◎</span>
          <span className="brand-name">GAUGE</span>
        </div>
        <span className="brand-tag">SaaS Churn Early-Warning Console</span>
        <nav className="tab-nav">
          <button className={`tab-btn ${tab === "predict" ? "tab-active" : ""}`} onClick={() => setTab("predict")}>
            Predict
          </button>
          <button className={`tab-btn ${tab === "portfolio" ? "tab-active" : ""}`} onClick={() => setTab("portfolio")}>
            Portfolio
          </button>
        </nav>
      </header>

      {tab === "predict" && (
        <>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Account health, instrumented</p>
          <h1>
            Read the churn signal <em>before</em> the renewal call.
          </h1>
          <p className="hero-sub">
            Feed in an account's usage, billing, and support signals. The model —
            trained on a synthetic portfolio of {modelInfo?.n_samples?.toLocaleString() || "6,000"} SaaS
            accounts — returns a calibrated churn probability and the exact factors driving it.
          </p>
          <div className="stat-strip">
            <div className="stat">
              <span className="stat-value">{modelInfo ? (modelInfo.churn_rate * 100).toFixed(1) + "%" : "—"}</span>
              <span className="stat-label">Portfolio churn rate</span>
            </div>
            <div className="stat">
              <span className="stat-value">
                {modelInfo ? (modelInfo.all_results[modelInfo.best_model].roc_auc * 100).toFixed(1) : "—"}
              </span>
              <span className="stat-label">Model ROC-AUC</span>
            </div>
            <div className="stat">
              <span className="stat-value">{modelInfo ? modelInfo.best_model.replace("_", " ") : "—"}</span>
              <span className="stat-label">Active model</span>
            </div>
          </div>
        </div>

        <div className="hero-gauge panel">
          {result && <Gauge probability={result.churn_probability} />}
          {result && (
            <div className={`risk-pill risk-${result.risk_tier.toLowerCase()}`}>
              {result.risk_tier} risk
            </div>
          )}
          <div className="preset-row">
            {Object.keys(PRESETS).map((name) => (
              <button key={name} type="button" className="preset-btn" onClick={() => applyPreset(name)}>
                {name}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="console">
        <form className="panel form-panel" onSubmit={onSubmit}>
          <h2>Account manifest</h2>
          <p className="panel-sub">Enter the account's current signals.</p>

          <div className="form-grid">
            <Field label="Plan tier">
              <select value={form.plan_tier} onChange={(e) => update("plan_tier", e.target.value)}>
                <option>Starter</option>
                <option>Growth</option>
                <option>Enterprise</option>
              </select>
            </Field>
            <Field label="Contract type">
              <select value={form.contract_type} onChange={(e) => update("contract_type", e.target.value)}>
                <option>Monthly</option>
                <option>Annual</option>
              </select>
            </Field>
            <Field label="Seats">
              <input type="number" min="1" value={form.seats} onChange={(e) => update("seats", +e.target.value)} />
            </Field>
            <Field label="Tenure (months)">
              <input type="number" min="0" value={form.tenure_months} onChange={(e) => update("tenure_months", +e.target.value)} />
            </Field>
            <Field label="Monthly price ($)">
              <input type="number" min="0" value={form.monthly_price} onChange={(e) => update("monthly_price", +e.target.value)} />
            </Field>
            <Field label="Discount applied (%)">
              <input type="number" min="0" max="100" value={form.discount_pct} onChange={(e) => update("discount_pct", +e.target.value)} />
            </Field>

            <Field label="Logins per seat / month">
              <input type="number" step="0.1" min="0" value={form.monthly_logins_per_seat} onChange={(e) => update("monthly_logins_per_seat", +e.target.value)} />
            </Field>
            <Field label="Days since last login">
              <input type="number" min="0" value={form.days_since_last_login} onChange={(e) => update("days_since_last_login", +e.target.value)} />
            </Field>
            <Field label={`Feature adoption: ${form.feature_adoption_pct}%`}>
              <input type="range" min="0" max="100" value={form.feature_adoption_pct} onChange={(e) => update("feature_adoption_pct", +e.target.value)} />
            </Field>
            <Field label="Integrations connected">
              <input type="number" min="0" value={form.integrations_connected} onChange={(e) => update("integrations_connected", +e.target.value)} />
            </Field>

            <Field label="Support tickets (90d)">
              <input type="number" min="0" value={form.support_tickets_last_90d} onChange={(e) => update("support_tickets_last_90d", +e.target.value)} />
            </Field>
            <Field label="Avg. ticket resolution (hrs)">
              <input type="number" min="0" value={form.avg_ticket_resolution_hours} onChange={(e) => update("avg_ticket_resolution_hours", +e.target.value)} />
            </Field>
            <Field label={`NPS score: ${form.nps_score}`}>
              <input type="range" min="-100" max="100" value={form.nps_score} onChange={(e) => update("nps_score", +e.target.value)} />
            </Field>

            <label className="field field-toggle">
              <input type="checkbox" checked={form.had_payment_failure_last_90d} onChange={(e) => update("had_payment_failure_last_90d", e.target.checked)} />
              <span>Payment failure in last 90 days</span>
            </label>
            <label className="field field-toggle">
              <input type="checkbox" checked={form.onboarding_completed} onChange={(e) => update("onboarding_completed", e.target.checked)} />
              <span>Onboarding completed</span>
            </label>
            <label className="field field-toggle">
              <input type="checkbox" checked={form.champion_left_company} onChange={(e) => update("champion_left_company", e.target.checked)} />
              <span>Internal champion left the company</span>
            </label>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Reading signal…" : "Run prediction"}
          </button>
          {error && <p className="error-text">Couldn't reach the model: {error}</p>}
        </form>

        <div className="panel drivers-panel">
          <h2>What's driving this</h2>
          <p className="panel-sub">
            Signed contribution of each signal to the churn probability, via SHAP.
          </p>
          <ul className="driver-list">
            {result?.top_drivers?.map((d) => {
              const magnitude = Math.min(Math.abs(d.impact) / 2, 1) * 100;
              const isRisk = d.impact > 0;
              return (
                <li key={d.feature} className="driver-row">
                  <div className="driver-meta">
                    <span className="driver-label">{d.label}</span>
                    <span className="driver-value">{d.value}</span>
                  </div>
                  <div className="driver-bar-track">
                    <div
                      className={`driver-bar ${isRisk ? "driver-bar-risk" : "driver-bar-safe"}`}
                      style={{ width: `${magnitude}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          {!result?.top_drivers?.length && <p className="panel-sub">Run a prediction to see the drivers.</p>}
        </div>
      </section>
      </>
      )}

      {tab === "portfolio" && <Dashboard />}

      <footer className="footer">
        <span>Gauge — a synthetic-data demo project. Not connected to a real customer database.</span>
      </footer>
    </div>
  );
}
