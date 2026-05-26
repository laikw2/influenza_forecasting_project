# Influenza Forecasting Experiment Summary

This experiment predicts next-week laboratory surveillance activity from WHO FluNet data for UAE and Malaysia.

## What Was Modeled
- Regression: next-week influenza A, influenza B, total influenza positives, positivity rate, A rate, and B rate.
- Classification: increase/stable/decrease trend, increase vs not-increase, and subtype driver.

## Key Limitation
FluNet reports laboratory surveillance results. These values are not complete national influenza incidence, and changes in laboratory/reporting coverage can change the signal.

## Outputs
- Cleaned data: `C:\Users\LAI\Documents\Codex\2026-05-25\please-check-this-website-link-https\influenza_forecasting_project\data`
- Models: `C:\Users\LAI\Documents\Codex\2026-05-25\please-check-this-website-link-https\influenza_forecasting_project\models`
- Reports: `C:\Users\LAI\Documents\Codex\2026-05-25\please-check-this-website-link-https\influenza_forecasting_project\reports`
- Plots: `C:\Users\LAI\Documents\Codex\2026-05-25\please-check-this-website-link-https\influenza_forecasting_project\reports\plots`

## Main Tables
- `model_comparison_metrics.csv`: all model validation metrics.
- `best_model_by_task.csv`: best model per scenario/task/target.
- `holdout_predictions.csv`: observed and predicted holdout values.
- `next_week_forecasts.csv`: latest next-week model forecasts for dashboard review.
- `skipped_models.csv`: optional or failed models with reasons.
- `checkpoints/`: scenario-level checkpoint CSVs and `*.done.json` files for safe resume.

## Resume Status
- Completed scenarios are loaded from `reports/checkpoints/` and skipped on rerun.
- Transfer scenarios are retained for regression comparison. Classification models are primarily evaluated in country-only and joint scenarios; transfer classification skips are recorded in `skipped_models.csv`.

## App
Run: `python -m streamlit run app/streamlit_app.py`