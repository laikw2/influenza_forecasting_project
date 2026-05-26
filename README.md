# Regional Transfer Learning Influenza Forecasting Dashboard

Numerical forecasting and cautious increase-alert prediction for UAE and Malaysia.

This project uses saved outputs from the WHO FluNet influenza forecasting experiment. The final dashboard focuses on the strongest regional transfer-learning models, not the full exploratory set of hundreds of model-target results.

## Recommended Dashboard Models

### UAE

- `next_INF_ALL`: Transfer UAE Random Forest
- `next_positivity_rate`: Transfer UAE Random Forest
- `next_INF_A`: Transfer UAE Random Forest
- `next_INF_B`: Transfer UAE Random Forest
- Increase Alert: Transfer UAE Random Forest, moderate confidence

### Malaysia

- `next_INF_ALL`: Transfer Malaysia Elastic Net
- `next_positivity_rate`: Transfer Malaysia LightGBM
- `next_INF_A`: Transfer Malaysia XGBoost
- `next_INF_B`: Transfer Malaysia Elastic Net
- Increase Alert: Transfer Malaysia LightGBM, low-to-moderate confidence

Malaysia increase-alert validation:

- Accuracy: 0.702
- Balanced accuracy: 0.542
- Macro F1: 0.502
- ROC-AUC: 0.641
- PR-AUC: 0.480
- Holdout weeks: 235

## Setup

From this folder:

```powershell
pip install -r requirements.txt
```

## Run The Final Dashboard

```powershell
streamlit run app.py
```

If Streamlit is only available through Python:

```powershell
python -m streamlit run app.py
```

## Main Files Used By The App

- `reports/next_week_forecasts.csv`
- `reports/model_comparison_metrics.csv`
- `reports/best_model_by_task.csv`
- `reports/plots/regression/*.png`
- `reports/transfer_uae_target_increase_binary_random_forest_confusion_matrix.csv`
- `reports/transfer_uae_target_increase_binary_random_forest_roc_curve.csv`
- `reports/transfer_uae_target_increase_binary_random_forest_pr_curve.csv`
- `reports/transfer_malaysia_target_increase_binary_lightgbm_confusion_matrix.csv`
- `reports/transfer_malaysia_target_increase_binary_lightgbm_roc_curve.csv`
- `reports/transfer_malaysia_target_increase_binary_lightgbm_pr_curve.csv`

The app also checks for the selected saved model files in `models/`.

## Scientific Interpretation

`INF_ALL` means total laboratory-confirmed influenza positives, combining influenza A and B. `positivity_rate` means total influenza positives divided by specimens processed when specimen count is positive.

The numerical regression forecasts are the strongest part of this project. The increase-alert classifiers are included as a practical warning layer, but should be interpreted only as low-to-moderate confidence. The subtype-driver and 3-class trend classifiers are not used as main dashboard outputs.

WHO FluNet data are laboratory surveillance reports. They should not be interpreted as complete national incidence. Reporting volume, laboratory participation, and surveillance practice can change over time. Prospective validation against future WHO FluNet updates is still required.

## Retraining

The final app does not retrain the full experiment. To reproduce or extend the experiment separately:

```powershell
python run_full_experiment.py
```

