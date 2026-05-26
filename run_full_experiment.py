import json
import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning

from src.config import DATA_DIR, MODELS_DIR, PLOTS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data_processing import prepare_core_datasets
from src.features import build_modeling_frame
from src.modeling import train_all_models


def write_final_summary(audit: dict, results: dict, model_df: pd.DataFrame, feature_cols: list[str]) -> None:
    metrics = pd.DataFrame(results["metrics"])
    skipped = pd.DataFrame(results["skipped"])
    best_rows = []
    if not metrics.empty:
        for (scenario, task, target), part in metrics.groupby(["scenario", "task", "target"]):
            if task == "regression" and "rmse" in part:
                best = part.sort_values("rmse", na_position="last").head(1)
            elif task == "classification" and "macro_f1" in part:
                best = part.sort_values("macro_f1", ascending=False, na_position="last").head(1)
            else:
                best = part.head(1)
            best_rows.extend(best.to_dict("records"))
    pd.DataFrame(best_rows).to_csv(REPORTS_DIR / "best_model_by_task.csv", index=False)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "data_audit": audit,
        "rows_in_modeling_frame": int(len(model_df)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "metric_rows": int(len(metrics)),
        "skipped_model_rows": int(len(skipped)),
        "important_interpretation": [
            "Regression metrics evaluate numerical next-week values.",
            "Classification metrics evaluate derived increase/stable/decrease and subtype-driver labels.",
            "ROC-AUC and PR-AUC are reported only for classification models with probabilities.",
            "FluNet is laboratory surveillance data and should not be interpreted as complete national incidence.",
        ],
        "external_features_status": {
            "weather": "placeholder supported; not included because no local weather file was provided",
            "google_trends": "placeholder supported; not included because no local trends file was provided",
            "holidays_mobility": "placeholder supported; not included because no local external file was provided",
        },
        "classical_time_series_baseline": "SARIMAX/Prophet enabled when statsmodels/prophet are installed.",
        "support_vector_models": "SVR/SVC intentionally omitted from the unmanned run because transfer-stage probability fitting was too slow for reliable unattended completion.",
        "neural_models": "PyTorch LSTM, GRU, and Temporal CNN enabled for UAE-only and Malaysia-only experiments when torch is installed.",
    }
    (REPORTS_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Influenza Forecasting Experiment Summary",
        "",
        "This experiment predicts next-week laboratory surveillance activity from WHO FluNet data for UAE and Malaysia.",
        "",
        "## What Was Modeled",
        "- Regression: next-week influenza A, influenza B, total influenza positives, positivity rate, A rate, and B rate.",
        "- Classification: increase/stable/decrease trend, increase vs not-increase, and subtype driver.",
        "",
        "## Key Limitation",
        "FluNet reports laboratory surveillance results. These values are not complete national influenza incidence, and changes in laboratory/reporting coverage can change the signal.",
        "",
        "## Outputs",
        f"- Cleaned data: `{DATA_DIR}`",
        f"- Models: `{MODELS_DIR}`",
        f"- Reports: `{REPORTS_DIR}`",
        f"- Plots: `{PLOTS_DIR}`",
        "",
        "## Main Tables",
        "- `model_comparison_metrics.csv`: all model validation metrics.",
        "- `best_model_by_task.csv`: best model per scenario/task/target.",
        "- `holdout_predictions.csv`: observed and predicted holdout values.",
        "- `next_week_forecasts.csv`: latest next-week model forecasts for dashboard review.",
        "- `skipped_models.csv`: optional or failed models with reasons.",
        "- `checkpoints/`: scenario-level checkpoint CSVs and `*.done.json` files for safe resume.",
        "",
        "## Resume Status",
        "- Completed scenarios are loaded from `reports/checkpoints/` and skipped on rerun.",
        "- Transfer scenarios are retained for regression comparison. Classification models are primarily evaluated in country-only and joint scenarios; transfer classification skips are recorded in `skipped_models.csv`.",
        "",
        "## App",
        "Run: `python -m streamlit run app/streamlit_app.py`",
    ]
    (REPORTS_DIR / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_core_datasets()
    target_frame, target_features = build_modeling_frame(prepared["target_weekly"])
    transfer_frame, transfer_features = build_modeling_frame(prepared["transfer_weekly"])
    feature_cols = sorted(set(target_features) & set(transfer_features))
    model_df = transfer_frame.copy()
    model_df.to_csv(DATA_DIR / "modeling_frame.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(DATA_DIR / "feature_columns.csv", index=False)

    results = train_all_models(model_df, feature_cols)
    write_final_summary(prepared["audit"], results, model_df, feature_cols)
    print("Experiment complete.")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Metrics: {REPORTS_DIR / 'model_comparison_metrics.csv'}")
    print(f"Summary: {REPORTS_DIR / 'final_summary.md'}")


if __name__ == "__main__":
    main()
