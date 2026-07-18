# Digester biogas GA optimizer (deploy-only)

Public Streamlit app; source can stay in a **private** GitHub repo.

## Contents
- `streamlit_app.py` — UI
- `ga_core.py` — GA + RF
- `weather_service.py` — Open-Meteo + CSV fallback
- `model_biogaz_rf_optimized.joblib` — trained model
- `weather_daily_api.csv` — weather fallback
- `.streamlit/config.toml`

## Deploy (private repo)

1. Create a **new private** GitHub repository (e.g. `sewage_cmes_app_deploy`).
2. Push only this folder as the repo root:

```bat
cd deploy_cloud
git init
git add .
git commit -m "Deploy-only Streamlit app"
git branch -M main
git remote add origin https://github.com/<USER>/<PRIVATE_REPO>.git
git push -u origin main
```

3. Go to https://share.streamlit.io → New app  
   - Repository: your **private** repo  
   - Branch: `main`  
   - Main file: `streamlit_app.py`  
4. Deploy. Share only the Streamlit URL (e.g. `https://xxx.streamlit.app`), not the GitHub repo.

Community Cloud: private GitHub source is OK; you get a public app link (or 1 fully private app on the free tier).

## Local test

```bat
cd deploy_cloud
..\.venv\Scripts\streamlit.exe run streamlit_app.py
```
