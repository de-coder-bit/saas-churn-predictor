import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { getPortfolioStats } from "./api";

const RISK_COLORS = { Low: "#6FBFB0", Medium: "#C9C15A", High: "#E8A33D", Critical: "#D9614F" };

function ChartCard({ title, sub, children }) {
  return (
    <div className="panel chart-card">
      <h3>{title}</h3>
      {sub && <p className="panel-sub">{sub}</p>}
      {children}
    </div>
  );
}

function TooltipBox({ active, payload, label, suffix = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="chart-tooltip-row">
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}{suffix}</strong>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getPortfolioStats().then(setStats).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error-text">Couldn't load portfolio stats: {error}</p>;
  if (!stats) return <p className="panel-sub">Loading portfolio…</p>;

  const planData = Object.entries(stats.churn_by_plan).map(([plan, rate]) => ({
    plan, "Predicted churn rate": +(rate * 100).toFixed(1),
  }));
  const contractData = Object.entries(stats.churn_by_contract).map(([type, rate]) => ({
    type, "Predicted churn rate": +(rate * 100).toFixed(1),
  }));
  const riskData = Object.entries(stats.risk_distribution).map(([tier, count]) => ({
    tier, count,
  }));
  const driverData = stats.global_drivers.map((d) => ({
    label: d.label, importance: d.importance,
  }));
  const cohortRows = Object.entries(stats.cohort_compare);

  return (
    <div className="dashboard">
      <div className="dash-grid">
        <ChartCard title="Churn rate by plan tier" sub="Model-predicted average churn probability per segment.">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={planData} margin={{ left: -10 }}>
              <CartesianGrid stroke="var(--hairline)" vertical={false} />
              <XAxis dataKey="plan" tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={{ stroke: "var(--hairline)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<TooltipBox suffix="%" />} cursor={{ fill: "var(--panel-raised)" }} />
              <Bar dataKey="Predicted churn rate" radius={[6, 6, 0, 0]}>
                {planData.map((_, i) => <Cell key={i} fill="var(--amber)" />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Churn rate by contract type" sub="Monthly vs. annual commitment.">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={contractData} margin={{ left: -10 }}>
              <CartesianGrid stroke="var(--hairline)" vertical={false} />
              <XAxis dataKey="type" tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={{ stroke: "var(--hairline)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<TooltipBox suffix="%" />} cursor={{ fill: "var(--panel-raised)" }} />
              <Bar dataKey="Predicted churn rate" radius={[6, 6, 0, 0]}>
                {contractData.map((_, i) => <Cell key={i} fill="var(--teal-signal)" />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Portfolio risk distribution" sub={`Across all ${stats.n_accounts.toLocaleString()} simulated accounts.`}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={riskData} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid stroke="var(--hairline)" horizontal={false} />
              <XAxis type="number" tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="tier" tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={false} tickLine={false} width={70} />
              <Tooltip content={<TooltipBox />} cursor={{ fill: "var(--panel-raised)" }} />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {riskData.map((d, i) => <Cell key={i} fill={RISK_COLORS[d.tier]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Global model drivers" sub="Which signals matter most across the whole portfolio.">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={driverData} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid stroke="var(--hairline)" horizontal={false} />
              <XAxis type="number" tick={{ fill: "var(--paper-dim)", fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="label" tick={{ fill: "var(--paper-dim)", fontSize: 11 }} axisLine={false} tickLine={false} width={140} />
              <Tooltip content={<TooltipBox />} cursor={{ fill: "var(--panel-raised)" }} />
              <Bar dataKey="importance" radius={[0, 6, 6, 0]} fill="var(--amber)" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="panel cohort-panel">
        <h3>Churned vs. retained: average signals</h3>
        <p className="panel-sub">How the two cohorts differ, on average, across the portfolio.</p>
        <table className="cohort-table">
          <thead>
            <tr><th>Signal</th><th>Churned accounts</th><th>Retained accounts</th></tr>
          </thead>
          <tbody>
            {cohortRows.map(([key, val]) => (
              <tr key={key}>
                <td>{key.replace(/_/g, " ")}</td>
                <td className="cohort-churned">{val.churned}</td>
                <td className="cohort-retained">{val.retained}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
