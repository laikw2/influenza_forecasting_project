import warnings
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_full_experiment import write_final_summary
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COUNTRIES
from src.data_processing import prepare_core_datasets
from src.evaluation import classification_metrics, save_confusion_matrix, save_curve_data
from src.features import build_modeling_frame
from src.modeling import (
    CATEGORICAL_FEATURES,
    PLOTS_DIR,
    base_classification_models,
    build_scenarios,
    country_holdout_split,
    future_forecast_records,
    make_preprocessor,
    model_path,
    prediction_records,
    read_checkpoint_csv,
    save_iterative_classification_loss,
    save_model_artifacts,
    scenario_checkpoint_paths,
    tune_pipeline,
    write_dependency_report,
    write_scenario_checkpoint,
)
from src.plotting import save_confusion_plot


SCENARIO_NAME = "transfer_malaysia"
CLASSIFICATION_TARGETS = ["target_increase_binary"]
ALLOWED_MODELS = {"random_forest", "lightgbm"}


def load_existing_transfer_rows() -> dict[str, list[dict]]:
    paths = scenario_checkpoint_paths(SCENARIO_NAME)
    rows = {
        "metrics": read_checkpoint_csv(paths["metrics"]),
        "skipped": read_checkpoint_csv(paths["skipped"]),
        "predictions": read_checkpoint_csv(paths["predictions"]),
        "future": read_checkpoint_csv(paths["future"]),
        "models": read_checkpoint_csv(paths["models"]),
    }
    for key in ["metrics", "predictions", "future", "models"]:
        rows[key] = [
            row
            for row in rows[key]
            if not (row.get("task") == "classification" and row.get("target") in CLASSIFICATION_TARGETS)
        ]
    rows["skipped"] = [
        row
        for row in rows["skipped"]
        if not (row.get("task") == "classification" and row.get("target") in CLASSIFICATION_TARGETS)
    ]
    return rows


def rebuild_final_tables(audit: dict, model_df: pd.DataFrame, feature_cols: list[str]) -> None:
    checkpoint_dir = REPORTS_DIR / "checkpoints"
    metrics, skipped, predictions, future, models = [], [], [], [], []
    for done_path in sorted(checkpoint_dir.glob("*.done.json")):
        scenario = done_path.name.replace(".done.json", "")
        paths = scenario_checkpoint_paths(scenario)
        metrics.extend(read_checkpoint_csv(paths["metrics"]))
        skipped.extend(read_checkpoint_csv(paths["skipped"]))
        predictions.extend(read_checkpoint_csv(paths["predictions"]))
        future.extend(read_checkpoint_csv(paths["future"]))
        models.extend(read_checkpoint_csv(paths["models"]))

    pd.DataFrame(metrics).to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    pd.DataFrame(skipped).to_csv(REPORTS_DIR / "skipped_models.csv", index=False)
    pd.DataFrame(predictions).to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)
    pd.DataFrame(future).to_csv(REPORTS_DIR / "next_week_forecasts.csv", index=False)
    pd.DataFrame(models).to_csv(MODELS_DIR / "model_registry.csv", index=False)
    write_dependency_report(skipped)
    write_final_summary(audit, {"metrics": metrics, "skipped": skipped}, model_df, feature_cols)


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

    prepared = prepare_core_datasets()
    target_frame, target_features = build_modeling_frame(prepared["target_weekly"])
    transfer_frame, transfer_features = build_modeling_frame(prepared["transfer_weekly"])
    feature_cols = sorted(set(target_features) & set(transfer_features))
    model_df = transfer_frame.copy()
    model_df.to_csv(DATA_DIR / "modeling_frame.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(DATA_DIR / "feature_columns.csv", index=False)

    scenario = next(item for item in build_scenarios() if item.name == SCENARIO_NAME)
    scenario_df = model_df[model_df["COUNTRY_CODE"].isin(scenario.training_countries)].copy()
    target_df = scenario_df[scenario_df["COUNTRY_CODE"].isin(scenario.target_countries)].copy()
    target_train, target_test = country_holdout_split(target_df, scenario.target_countries)
    pool_train = scenario_df[~scenario_df["COUNTRY_CODE"].isin(set(TARGET_COUNTRIES))].copy()
    train_df = pd.concat([pool_train, target_train], ignore_index=True).sort_values(["ISO_WEEKSTARTDATE", "COUNTRY_CODE"])
    test_df = target_test

    existing = load_existing_transfer_rows()
    metric_rows = existing["metrics"]
    skipped_rows = existing["skipped"]
    prediction_rows = existing["predictions"]
    future_rows = existing["future"]
    model_registry = existing["models"]

    for target in CLASSIFICATION_TARGETS:
        clean_train = train_df.dropna(subset=[target])
        clean_test = test_df.dropna(subset=[target])
        if len(clean_train) < 40 or len(clean_test) < 5 or clean_train[target].nunique() < 2:
            skipped_rows.append({"scenario": SCENARIO_NAME, "task": "classification", "target": target, "model": "all", "reason": "not enough rows or classes"})
            continue

        for model_name, (estimator, grid) in base_classification_models().items():
            if model_name not in ALLOWED_MODELS:
                continue
            pipe = Pipeline([("preprocessor", make_preprocessor(feature_cols)), ("model", estimator)])
            X_train = clean_train[feature_cols + CATEGORICAL_FEATURES]
            y_train = clean_train[target].astype(str)
            X_test = clean_test[feature_cols + CATEGORICAL_FEATURES]
            y_test = clean_test[target].astype(str)
            try:
                best, info, cv_rows = tune_pipeline(pipe, grid, X_train, y_train, "classification")
                pred = best.predict(X_test)
                proba = best.predict_proba(X_test) if hasattr(best, "predict_proba") else None
                classes = list(best.classes_) if hasattr(best, "classes_") else sorted(y_train.unique())
                metrics = classification_metrics(y_test, pred, proba, classes)
                metric_rows.append({"scenario": SCENARIO_NAME, "task": "classification", "target": target, "model": model_name, **metrics})
                save_model_artifacts(best, SCENARIO_NAME, "classification", target, model_name, info, cv_rows)
                cm_path = REPORTS_DIR / f"{SCENARIO_NAME}_{target}_{model_name}_confusion_matrix.csv"
                cm_df = save_confusion_matrix(y_test, pred, classes, cm_path)
                save_confusion_plot(cm_df, PLOTS_DIR / "classification" / f"{SCENARIO_NAME}_{target}_{model_name}_confusion_matrix.png", f"{SCENARIO_NAME} {target} {model_name}")
                if proba is not None:
                    save_curve_data(y_test, proba, classes, REPORTS_DIR / f"{SCENARIO_NAME}_{target}_{model_name}")
                if model_name == "lightgbm":
                    save_iterative_classification_loss(best, X_train, y_train, X_test, y_test, SCENARIO_NAME, target, model_name)
                prediction_rows.extend(prediction_records(SCENARIO_NAME, "classification", target, model_name, clean_test, y_test, pred))
                future_rows.extend(future_forecast_records(SCENARIO_NAME, "classification", target, model_name, best, model_df, scenario.target_countries, feature_cols, None))
                model_registry.append({"scenario": SCENARIO_NAME, "task": "classification", "target": target, "model": model_name, "artifact": str(model_path(SCENARIO_NAME, "classification", target, model_name))})
            except Exception as exc:
                skipped_rows.append({"scenario": SCENARIO_NAME, "task": "classification", "target": target, "model": model_name, "reason": repr(exc)})

    skipped_rows = [
        row
        for row in skipped_rows
        if not (row.get("task") == "classification" and row.get("target") == "target_subtype_driver")
    ]
    skipped_rows.append(
        {
            "scenario": SCENARIO_NAME,
            "task": "classification",
            "target": "target_subtype_driver",
            "model": "all",
            "reason": "skipped in transfer Malaysia because subtype-driver transfer classification is exploratory and slow; country-only and joint subtype-driver models are available",
        }
    )
    skipped_rows.append(
        {
            "scenario": SCENARIO_NAME,
            "task": "classification",
            "target": "target_trend",
            "model": "all",
            "reason": "not used as transfer Malaysia app classifier in this targeted run; train separately if a 3-class transfer trend model is required",
        }
    )

    write_scenario_checkpoint(SCENARIO_NAME, metric_rows, skipped_rows, prediction_rows, future_rows, model_registry)
    rebuild_final_tables(prepared["audit"], model_df, feature_cols)
    print("Transfer Malaysia classification training complete.")


if __name__ == "__main__":
    main()
