# Streamlit Deployment Notes

## Local Use

Run the full experiment first:

```powershell
python run_full_experiment.py
```

Then launch the app:

```powershell
python -m streamlit run app/streamlit_app.py
```

## Files Needed For The App

- `app/streamlit_app.py`
- `data/target_country_weekly_clean.csv`
- `reports/model_comparison_metrics.csv`
- `reports/best_model_by_task.csv`
- `reports/holdout_predictions.csv`
- `reports/next_week_forecasts.csv`
- `reports/skipped_models.csv`
- `reports/plots/`
- `requirements.txt`

## Cloud Deployment

For Streamlit Community Cloud, upload this project to GitHub after the experiment outputs are generated, or adapt the app to download FluNet data during startup. For collaborator testing, the simplest route is to share the generated project folder or deploy a GitHub repo containing the saved outputs.
