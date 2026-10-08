# Gauge (Streamlit edition) — SaaS Churn Console

Same model, same design language, but running as a single Streamlit process —
one server, one command, no CORS/networking setup required. Good option if
you hit local networking issues running the separate FastAPI + React version.

## Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open your browser automatically to `http://localhost:8501`.
That's it — no second terminal, no `.env` file, no CORS.

## What's inside

- `app.py` — the whole app: model loading, prediction UI with a Plotly gauge,
  SHAP-based driver breakdown, and a Portfolio dashboard tab with churn-by-segment
  charts, global driver importances, and a churned-vs-retained cohort table.
- `model/` — the same trained artifacts used by the FastAPI backend
  (`churn_model.joblib`, `feature_names.joblib`, `metrics.json`,
  `saas_churn_dataset.csv`) — copied over so this folder is fully self-contained.
- `.streamlit/config.toml` — theme matching the rest of the project (dark
  teal background, amber accent).

## Deploy for free

**Streamlit Community Cloud** (easiest):
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. New app → pick the repo → set the app file path to `streamlit_app/app.py`.
4. Deploy. You get a free `*.streamlit.app` URL.

That's the entire deployment process — no separate frontend/backend hosting,
no environment variables needed.

## Notes

- This reuses the exact same trained model as the FastAPI/React version — same
  predictions, same calibration. If you retrain the model (see the `backend/`
  folder's `train.py`), copy the four updated files from
  `backend/app/model/` into `streamlit_app/model/` to keep both versions in sync.
