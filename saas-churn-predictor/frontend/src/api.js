const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictChurn(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) throw new Error("Could not load model info");
  return res.json();
}

export async function getPortfolioStats() {
  const res = await fetch(`${API_BASE}/portfolio-stats`);
  if (!res.ok) throw new Error("Could not load portfolio stats");
  return res.json();
}
