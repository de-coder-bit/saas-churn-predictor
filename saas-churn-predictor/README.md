# Gauge — SaaS Churn Early-Warning Console

A full-stack churn prediction app: synthetic SaaS account data → trained
classifier → FastAPI backend → React frontend with SHAP-explained
predictions and an instrument-panel UI.

```
saas-churn-predictor/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app: /predict, /model-info, /health
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── model/
│   │       ├── generate_data.py   # synthetic SaaS churn dataset generator
│   │       ├── train.py            # trains + compares 3 models, exports artifact
│   │       ├── saas_churn_dataset.csv
│   │       ├── churn_model.joblib  # trained sklearn Pipeline (committed)
│   │       ├── feature_names.joblib
│   │       └── metrics.json
│   ├── requirements.txt
│   ├── render.yaml            # Render free-tier deploy config
│   └── Procfile               # alt deploy config (Railway/Heroku-style)
└── frontend/
    ├── src/
    │   ├── App.jsx             # main UI: hero, prediction form, driver breakdown
    │   ├── Gauge.jsx            # custom SVG radial risk gauge
    │   ├── api.js               # backend API client
    │   └── index.css / App.css  # design tokens + styles
    └── vercel.json
```

## How it works

1. **Data**: `generate_data.py` simulates 6,000 SaaS accounts (plan tier, seats,
   tenure, usage, support tickets, NPS, payment failures, etc.) with churn
   generated from a logistic function of real drivers + noise — so the
   signal is realistic, not trivially separable.
2. **Model**: `train.py` compares Logistic Regression, Random Forest, and
   XGBoost (all with class-imbalance handling), picks the best by ROC-AUC
   (currently Logistic Regression, ~0.86 AUC), and exports the full
   preprocessing + model pipeline as one artifact.
3. **API**: FastAPI loads the artifact once at startup, and for every
   `/predict` call also runs a SHAP explainer so the response includes the
   top 5 signed feature contributions — not just a probability.
4. **UI**: React app — a radial gauge (the signature visual element) shows
   the live probability, a form lets you edit any account signal, and a
   driver panel shows what's pushing risk up or down, styled as a data
   "manifest."

## Run locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
# (data + model artifacts are already committed — only re-run these if you
# want to regenerate them)
python app/model/generate_data.py
python app/model/train.py
uvicorn app.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```
Open the printed localhost URL (default http://localhost:5173).

## Deploy for free

**Backend → Render**
1. Push this repo to GitHub.
2. On [render.com](https://render.com), New → Web Service → connect the repo,
   set root directory to `backend`. Render will read `render.yaml`
   automatically (free plan, Python runtime, correct build/start commands).
3. Note the deployed URL, e.g. `https://saas-churn-api.onrender.com`.
   (Free tier spins down after inactivity — first request after idle takes
   ~30–50s to wake up.)

**Frontend → Vercel**
1. On [vercel.com](https://vercel.com), New Project → import the repo, set
   root directory to `frontend`. Vercel auto-detects Vite.
2. Add an environment variable `VITE_API_URL` = your Render backend URL
   (from the step above).
3. Deploy. Vercel gives you a free `*.vercel.app` URL.

**Alternative all-in-one free options**: Hugging Face Spaces (Docker SDK) for
the backend, or Railway's free trial credits for either service.

## Notes

- CORS on the backend is currently open (`allow_origins=["*"]`) for ease of
  setup — tighten to your Vercel domain before treating this as production.
- The dataset is synthetic; swap in a real dataset by matching the column
  names in `train.py`'s `CATEGORICAL`/`NUMERIC` lists and retraining.
