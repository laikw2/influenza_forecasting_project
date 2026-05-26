import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CLASSIFICATION_TARGETS,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    REGRESSION_TARGETS,
    REPORTS_DIR,
    TARGET_COUNTRIES,
    TRANSFER_POOLS,
)
from .evaluation import (
    classification_metrics,
    regression_metrics,
    save_confusion_matrix,
    save_curve_data,
    write_json,
    add_interval_metrics,
)
from .plotting import (
    save_confusion_plot,
    save_loss_curve,
    save_observed_predicted_plot,
    save_residual_plot,
    save_calibration_plot,
)
from .statistical_models import statistical_metric_row
from .neural_models import train_sequence_model, torch_available


CATEGORICAL_FEATURES = ["COUNTRY_CODE", "WHOREGION", "ITZ", "HEMISPHERE"]
NEURAL_REGRESSION_TARGETS = {"target_next_INF_A", "target_next_INF_B", "target_next_INF_ALL"}
NEURAL_CLASSIFICATION_TARGETS = {"target_trend", "target_subtype_driver"}
NEURAL_SCENARIOS = {"uae_only", "malaysia_only"}
NEURAL_ARCHITECTURES = ["lstm", "gru", "temporal_cnn"]
CHECKPOINT_DIR = REPORTS_DIR / "checkpoints"


@dataclass
class Scenario:
    name: str
    target_countries: list[str]
    training_countries: list[str]
    description: str


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_scenarios() -> list[Scenario]:
    return [
        Scenario("uae_only", ["ARE"], ["ARE"], "UAE model trained only on UAE FluNet weeks."),
        Scenario("malaysia_only", ["MYS"], ["MYS"], "Malaysia model trained only on Malaysia FluNet weeks."),
        Scenario("joint_uae_malaysia", ["ARE", "MYS"], ["ARE", "MYS"], "Joint UAE and Malaysia model."),
        Scenario("transfer_uae", ["ARE"], sorted(set(["ARE"] + TRANSFER_POOLS["ARE"])), "UAE target model with West/South Asia pretraining pool."),
        Scenario("transfer_malaysia", ["MYS"], sorted(set(["MYS"] + TRANSFER_POOLS["MYS"])), "Malaysia target model with Southeast Asia pretraining pool."),
    ]


def country_holdout_split(df: pd.DataFrame, target_countries: list[str], holdout_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for country in target_countries:
        part = df[df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        if len(part) < 30:
            continue
        cut = max(1, int(np.floor(len(part) * (1 - holdout_frac))))
        train_parts.append(part.iloc[:cut])
        test_parts.append(part.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_features = feature_cols
    categorical_features = CATEGORICAL_FEATURES
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def base_regression_models() -> dict:
    models = {
        "elastic_net": (
            ElasticNet(max_iter=5000, random_state=RANDOM_STATE),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.5]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100],
                "model__max_depth": [4, None],
                "model__min_samples_leaf": [3],
            },
        ),
    }
    if optional_module("xgboost"):
        from xgboost import XGBRegressor

        models["xgboost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="rmse",
            ),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05],
                "model__max_depth": [3],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        )
    if optional_module("lightgbm"):
        from lightgbm import LGBMRegressor

        models["lightgbm"] = (
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05],
                "model__num_leaves": [15],
                "model__min_child_samples": [20],
            },
        )
    return models


def base_classification_models() -> dict:
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.5, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
            {
                "model__n_estimators": [100],
                "model__max_depth": [4, None],
                "model__min_samples_leaf": [3],
            },
        ),
    }
    if optional_module("lightgbm"):
        from lightgbm import LGBMClassifier

        models["lightgbm"] = (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05],
                "model__num_leaves": [15],
                "model__min_child_samples": [20],
            },
        )
    return models


def tune_pipeline(pipe: Pipeline, param_grid: dict, X: pd.DataFrame, y: pd.Series, task: str) -> tuple[Pipeline, dict, list[dict]]:
    n_splits = min(3, max(2, len(X) // 120))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = np.inf
    best_params = {}
    cv_rows = []
    for params in ParameterGrid(param_grid):
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            candidate = Pipeline(pipe.steps)
            candidate.set_params(**params)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if y_train.nunique(dropna=True) < 2 and task == "classification":
                continue
            candidate.fit(X_train, y_train)
            pred = candidate.predict(X_val)
            if task == "regression":
                score = float(np.sqrt(mean_squared_error(y_val, pred)))
            else:
                if hasattr(candidate, "predict_proba"):
                    proba = candidate.predict_proba(X_val)
                    score = log_loss(y_val, proba, labels=candidate.classes_)
                else:
                    score = 1 - np.mean(pred == y_val)
            scores.append(score)
        mean_score = float(np.mean(scores)) if scores else np.inf
        row = {"mean_cv_score": mean_score, **params}
        cv_rows.append(row)
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    best = Pipeline(pipe.steps)
    best.set_params(**best_params)
    best.fit(X, y)
    return best, {"best_cv_score": best_score, "best_params": best_params}, cv_rows


def simple_baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, mode: str) -> np.ndarray:
    all_parts = []
    for country, part in test_df.groupby("COUNTRY_CODE"):
        hist = train_df[train_df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        test_part = part.sort_values("ISO_WEEKSTARTDATE")
        history = list(hist[target_col.replace("target_next_", "")].dropna())
        preds = []
        for _, row in test_part.iterrows():
            if mode == "seasonal_naive" and len(history) >= 52:
                pred = history[-52]
            elif mode == "moving_average":
                pred = float(np.mean(history[-4:])) if history else np.nan
            else:
                pred = history[-1] if history else np.nan
            preds.append(pred)
            actual_current_signal = row.get(target_col.replace("target_next_", ""), np.nan)
            if pd.notna(actual_current_signal):
                history.append(float(actual_current_signal))
        all_parts.append(pd.Series(preds, index=test_part.index))
    return pd.concat(all_parts).sort_index().reindex(test_df.index).to_numpy()


def train_all_models(model_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "regression").mkdir(exist_ok=True)
    (PLOTS_DIR / "classification").mkdir(exist_ok=True)
    (PLOTS_DIR / "loss_curves").mkdir(exist_ok=True)

    all_metric_rows = []
    skipped_rows = []
    prediction_rows = []
    future_rows = []
    model_registry = []
    scenarios = build_scenarios()
    target_only = set(TARGET_COUNTRIES)

    for scenario in scenarios:
        scenario_checkpoint = load_scenario_checkpoint(scenario.name)
        if scenario_checkpoint is not None:
            all_metric_rows.extend(scenario_checkpoint["metrics"])
            skipped_rows.extend(scenario_checkpoint["skipped"])
            prediction_rows.extend(scenario_checkpoint["predictions"])
            future_rows.extend(scenario_checkpoint["future"])
            model_registry.extend(scenario_checkpoint["models"])
            continue

        scenario_metric_rows = []
        scenario_skipped_rows = []
        scenario_prediction_rows = []
        scenario_future_rows = []
        scenario_model_registry = []

        scenario_df = model_df[model_df["COUNTRY_CODE"].isin(scenario.training_countries)].copy()
        target_df = scenario_df[scenario_df["COUNTRY_CODE"].isin(scenario.target_countries)].copy()
        target_train, target_test = country_holdout_split(target_df, scenario.target_countries)
        if scenario.name.startswith("transfer_"):
            pool_train = scenario_df[~scenario_df["COUNTRY_CODE"].isin(target_only)].copy()
            train_df = pd.concat([pool_train, target_train], ignore_index=True).sort_values(["ISO_WEEKSTARTDATE", "COUNTRY_CODE"])
        else:
            train_df = target_train
        test_df = target_test

        for target in REGRESSION_TARGETS:
            clean_train = train_df.dropna(subset=[target])
            clean_test = test_df.dropna(subset=[target])
            if len(clean_train) < 40 or len(clean_test) < 5:
                scenario_skipped_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": "all", "reason": "not enough non-missing rows"})
                continue
            for baseline in ["seasonal_naive", "moving_average"]:
                pred = simple_baseline_predictions(clean_train, clean_test, target, baseline)
                residual_std = baseline_residual_std(clean_train, target, baseline)
                metrics = add_interval_metrics(regression_metrics(clean_test[target], pred), clean_test[target], pred, residual_std)
                scenario_metric_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": baseline, **metrics})
                scenario_prediction_rows.extend(prediction_records(scenario.name, "regression", target, baseline, clean_test, clean_test[target], pred))

            for stat_model in ["sarimax", "prophet"]:
                try:
                    pred, residual_std, metrics = statistical_metric_row(clean_train, clean_test, target, stat_model)
                    scenario_metric_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": stat_model, **metrics})
                    pred_df = pd.DataFrame(prediction_records(scenario.name, "regression", target, stat_model, clean_test, clean_test[target], pred))
                    scenario_prediction_rows.extend(pred_df.to_dict("records"))
                    plot_prefix = f"{scenario.name}_{target}_{stat_model}"
                    save_observed_predicted_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_observed_predicted.png", plot_prefix)
                    save_residual_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_residuals.png", f"Residuals: {plot_prefix}")
                    save_calibration_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_calibration.png", f"Calibration: {plot_prefix}")
                    scenario_future_rows.extend(statistical_future_records(scenario.name, target, stat_model, clean_train, model_df, scenario.target_countries, residual_std))
                except Exception as exc:
                    scenario_skipped_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": stat_model, "reason": repr(exc)})

            for model_name, (estimator, grid) in base_regression_models().items():
                pipe = Pipeline([("preprocessor", make_preprocessor(feature_cols)), ("model", estimator)])
                X_train = clean_train[feature_cols + CATEGORICAL_FEATURES]
                y_train = clean_train[target]
                X_test = clean_test[feature_cols + CATEGORICAL_FEATURES]
                y_test = clean_test[target]
                try:
                    best, info, cv_rows = tune_pipeline(pipe, grid, X_train, y_train, "regression")
                    pred = best.predict(X_test)
                    train_pred = best.predict(X_train)
                    residual_std = float(np.nanstd(y_train.to_numpy(dtype=float) - train_pred))
                    metrics = add_interval_metrics(regression_metrics(y_test, pred), y_test, pred, residual_std)
                    scenario_metric_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": model_name, **metrics})
                    save_model_artifacts(best, scenario.name, "regression", target, model_name, info, cv_rows)
                    if model_name in {"xgboost", "lightgbm"}:
                        save_iterative_regression_loss(best, X_train, y_train, X_test, y_test, scenario.name, target, model_name)
                    pred_df = pd.DataFrame(prediction_records(scenario.name, "regression", target, model_name, clean_test, y_test, pred))
                    scenario_prediction_rows.extend(pred_df.to_dict("records"))
                    scenario_future_rows.extend(future_forecast_records(scenario.name, "regression", target, model_name, best, model_df, scenario.target_countries, feature_cols, residual_std))
                    plot_prefix = f"{scenario.name}_{target}_{model_name}"
                    save_observed_predicted_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_observed_predicted.png", plot_prefix)
                    save_residual_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_residuals.png", f"Residuals: {plot_prefix}")
                    save_calibration_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_calibration.png", f"Calibration: {plot_prefix}")
                    scenario_model_registry.append({"scenario": scenario.name, "task": "regression", "target": target, "model": model_name, "artifact": str(model_path(scenario.name, "regression", target, model_name))})
                except Exception as exc:
                    scenario_skipped_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": model_name, "reason": repr(exc)})

            if target in NEURAL_REGRESSION_TARGETS and scenario.name in NEURAL_SCENARIOS:
                for arch in NEURAL_ARCHITECTURES:
                    try:
                        out_prefix = MODELS_DIR / f"{scenario.name}_regression_{target}_{arch}"
                        metrics, meta_test, y_true, y_pred, residual_std = train_sequence_model(
                            clean_train,
                            clean_test,
                            feature_cols,
                            target,
                            arch,
                            "regression",
                            PLOTS_DIR / "loss_curves" / f"{scenario.name}_{target}_{arch}",
                        )
                        scenario_metric_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": arch, **metrics})
                        neural_path = (PLOTS_DIR / "loss_curves" / f"{scenario.name}_{target}_{arch}").with_suffix(".joblib")
                        if neural_path.exists():
                            neural_path.replace(out_prefix.with_suffix(".joblib"))
                        pred_df = pd.DataFrame(prediction_records(scenario.name, "regression", target, arch, meta_test, y_true, y_pred))
                        scenario_prediction_rows.extend(pred_df.to_dict("records"))
                        plot_prefix = f"{scenario.name}_{target}_{arch}"
                        save_observed_predicted_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_observed_predicted.png", plot_prefix)
                        save_residual_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_residuals.png", f"Residuals: {plot_prefix}")
                        save_calibration_plot(pred_df, PLOTS_DIR / "regression" / f"{plot_prefix}_calibration.png", f"Calibration: {plot_prefix}")
                        scenario_model_registry.append({"scenario": scenario.name, "task": "regression", "target": target, "model": arch, "artifact": str(out_prefix.with_suffix(".joblib"))})
                    except Exception as exc:
                        scenario_skipped_rows.append({"scenario": scenario.name, "task": "regression", "target": target, "model": arch, "reason": repr(exc)})

        for target in CLASSIFICATION_TARGETS:
            if scenario.name.startswith("transfer_") and target == "target_subtype_driver":
                scenario_skipped_rows.append(
                    {
                        "scenario": scenario.name,
                        "task": "classification",
                        "target": target,
                        "model": "all",
                        "reason": "skipped in transfer scenarios because subtype-driver transfer classification is exploratory and slow; country-only and joint subtype-driver models are available",
                    }
                )
                continue
            clean_train = train_df.dropna(subset=[target])
            clean_test = test_df.dropna(subset=[target])
            if len(clean_train) < 40 or len(clean_test) < 5 or clean_train[target].nunique() < 2:
                scenario_skipped_rows.append({"scenario": scenario.name, "task": "classification", "target": target, "model": "all", "reason": "not enough rows or classes"})
                continue
            for model_name, (estimator, grid) in base_classification_models().items():
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
                    scenario_metric_rows.append({"scenario": scenario.name, "task": "classification", "target": target, "model": model_name, **metrics})
                    save_model_artifacts(best, scenario.name, "classification", target, model_name, info, cv_rows)
                    cm_path = REPORTS_DIR / f"{scenario.name}_{target}_{model_name}_confusion_matrix.csv"
                    cm_df = save_confusion_matrix(y_test, pred, classes, cm_path)
                    save_confusion_plot(cm_df, PLOTS_DIR / "classification" / f"{scenario.name}_{target}_{model_name}_confusion_matrix.png", f"{scenario.name} {target} {model_name}")
                    curve_prefix = REPORTS_DIR / f"{scenario.name}_{target}_{model_name}"
                    if proba is not None:
                        save_curve_data(y_test, proba, classes, curve_prefix)
                    if model_name in {"xgboost", "lightgbm"}:
                        save_iterative_classification_loss(best, X_train, y_train, X_test, y_test, scenario.name, target, model_name)
                    scenario_prediction_rows.extend(prediction_records(scenario.name, "classification", target, model_name, clean_test, y_test, pred))
                    scenario_future_rows.extend(future_forecast_records(scenario.name, "classification", target, model_name, best, model_df, scenario.target_countries, feature_cols, None))
                    scenario_model_registry.append({"scenario": scenario.name, "task": "classification", "target": target, "model": model_name, "artifact": str(model_path(scenario.name, "classification", target, model_name))})
                except Exception as exc:
                    scenario_skipped_rows.append({"scenario": scenario.name, "task": "classification", "target": target, "model": model_name, "reason": repr(exc)})

            if target in NEURAL_CLASSIFICATION_TARGETS and scenario.name in NEURAL_SCENARIOS:
                for arch in NEURAL_ARCHITECTURES:
                    try:
                        out_prefix = MODELS_DIR / f"{scenario.name}_classification_{target}_{arch}"
                        metrics, meta_test, y_true, y_pred, _ = train_sequence_model(
                            clean_train,
                            clean_test,
                            feature_cols,
                            target,
                            arch,
                            "classification",
                            PLOTS_DIR / "loss_curves" / f"{scenario.name}_{target}_{arch}",
                        )
                        scenario_metric_rows.append({"scenario": scenario.name, "task": "classification", "target": target, "model": arch, **metrics})
                        neural_path = (PLOTS_DIR / "loss_curves" / f"{scenario.name}_{target}_{arch}").with_suffix(".joblib")
                        if neural_path.exists():
                            neural_path.replace(out_prefix.with_suffix(".joblib"))
                        pred_df = pd.DataFrame(prediction_records(scenario.name, "classification", target, arch, meta_test, y_true, y_pred))
                        scenario_prediction_rows.extend(pred_df.to_dict("records"))
                        cm_path = REPORTS_DIR / f"{scenario.name}_{target}_{arch}_confusion_matrix.csv"
                        labels = sorted(pd.Series(y_true).astype(str).unique())
                        cm_df = save_confusion_matrix(y_true, y_pred, labels, cm_path)
                        save_confusion_plot(cm_df, PLOTS_DIR / "classification" / f"{scenario.name}_{target}_{arch}_confusion_matrix.png", f"{scenario.name} {target} {arch}")
                        scenario_model_registry.append({"scenario": scenario.name, "task": "classification", "target": target, "model": arch, "artifact": str(out_prefix.with_suffix(".joblib"))})
                    except Exception as exc:
                        scenario_skipped_rows.append({"scenario": scenario.name, "task": "classification", "target": target, "model": arch, "reason": repr(exc)})

        write_scenario_checkpoint(
            scenario.name,
            scenario_metric_rows,
            scenario_skipped_rows,
            scenario_prediction_rows,
            scenario_future_rows,
            scenario_model_registry,
        )
        all_metric_rows.extend(scenario_metric_rows)
        skipped_rows.extend(scenario_skipped_rows)
        prediction_rows.extend(scenario_prediction_rows)
        future_rows.extend(scenario_future_rows)
        model_registry.extend(scenario_model_registry)

    pd.DataFrame(all_metric_rows).to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(REPORTS_DIR / "skipped_models.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)
    pd.DataFrame(future_rows).to_csv(REPORTS_DIR / "next_week_forecasts.csv", index=False)
    pd.DataFrame(model_registry).to_csv(MODELS_DIR / "model_registry.csv", index=False)
    write_dependency_report(skipped_rows)
    return {
        "metrics": all_metric_rows,
        "skipped": skipped_rows,
        "models": model_registry,
    }


def baseline_residual_std(train_df: pd.DataFrame, target_col: str, mode: str) -> float:
    rows = []
    for country, part in train_df.groupby("COUNTRY_CODE"):
        part = part.sort_values("ISO_WEEKSTARTDATE")
        values = part[target_col.replace("target_next_", "")].to_list()
        y = part[target_col].to_numpy(dtype=float)
        preds = []
        for i in range(len(part)):
            hist = [v for v in values[: i + 1] if pd.notna(v)]
            if mode == "seasonal_naive" and len(hist) >= 52:
                preds.append(hist[-52])
            elif mode == "moving_average" and hist:
                preds.append(float(np.mean(hist[-4:])))
            else:
                preds.append(hist[-1] if hist else np.nan)
        resid = y - np.asarray(preds, dtype=float)
        rows.extend(resid[np.isfinite(resid)])
    return float(np.nanstd(rows)) if rows else np.nan


def future_forecast_records(scenario, task, target, model_name, model, model_df, target_countries, feature_cols, residual_std) -> list[dict]:
    rows = []
    needed = feature_cols + CATEGORICAL_FEATURES
    for country in target_countries:
        part = model_df[model_df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        available = part.dropna(subset=needed, how="all")
        if available.empty:
            continue
        row = available.iloc[[-1]]
        try:
            pred = model.predict(row[needed])[0]
            rec = {
                "scenario": scenario,
                "task": task,
                "target": target,
                "model": model_name,
                "COUNTRY_CODE": country,
                "latest_observed_week": row["ISO_WEEKSTARTDATE"].iloc[0],
                "forecast_week": row["ISO_WEEKSTARTDATE"].iloc[0] + pd.Timedelta(days=7),
                "prediction": pred,
            }
            if task == "regression" and residual_std and np.isfinite(residual_std):
                rec["lower_95"] = float(pred - 1.96 * residual_std)
                rec["upper_95"] = float(pred + 1.96 * residual_std)
                rec["lower_80"] = float(pred - 1.28 * residual_std)
                rec["upper_80"] = float(pred + 1.28 * residual_std)
            if task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(row[needed])[0]
                for cls, p in zip(model.classes_, proba):
                    rec[f"prob_{cls}"] = float(p)
            rows.append(rec)
        except Exception:
            continue
    return rows


def prediction_records(scenario, task, target, model, df, y_true, y_pred) -> list[dict]:
    rows = []
    df = df.reset_index(drop=True)
    for pos, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        row = df.iloc[pos]
        rows.append(
            {
                "scenario": scenario,
                "task": task,
                "target": target,
                "model": model,
                "COUNTRY_CODE": row["COUNTRY_CODE"],
                "ISO_WEEKSTARTDATE": row["ISO_WEEKSTARTDATE"],
                "y_true": true_value,
                "y_pred": pred_value,
            }
        )
    return rows


def statistical_future_records(scenario, target, model_name, train_df, model_df, target_countries, residual_std) -> list[dict]:
    rows = []
    signal_col = target.replace("target_next_", "")
    for country in target_countries:
        part = model_df[model_df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        latest = part.dropna(subset=[signal_col]).tail(1)
        if latest.empty:
            continue
        fake_test = latest.copy()
        fake_test[target] = latest[signal_col].to_numpy()
        try:
            pred, _, _ = statistical_metric_row(train_df, fake_test, target, model_name)
            rec = {
                "scenario": scenario,
                "task": "regression",
                "target": target,
                "model": model_name,
                "COUNTRY_CODE": country,
                "latest_observed_week": latest["ISO_WEEKSTARTDATE"].iloc[0],
                "forecast_week": latest["ISO_WEEKSTARTDATE"].iloc[0] + pd.Timedelta(days=7),
                "prediction": float(pred[0]),
            }
            if np.isfinite(residual_std):
                rec["lower_95"] = float(pred[0] - 1.96 * residual_std)
                rec["upper_95"] = float(pred[0] + 1.96 * residual_std)
                rec["lower_80"] = float(pred[0] - 1.28 * residual_std)
                rec["upper_80"] = float(pred[0] + 1.28 * residual_std)
            rows.append(rec)
        except Exception:
            continue
    return rows


def scenario_checkpoint_paths(scenario_name: str) -> dict[str, Path]:
    prefix = CHECKPOINT_DIR / scenario_name
    return {
        "metrics": prefix.with_suffix(".metrics.csv"),
        "skipped": prefix.with_suffix(".skipped.csv"),
        "predictions": prefix.with_suffix(".predictions.csv"),
        "future": prefix.with_suffix(".future.csv"),
        "models": prefix.with_suffix(".models.csv"),
        "done": prefix.with_suffix(".done.json"),
    }


def read_checkpoint_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def load_scenario_checkpoint(scenario_name: str) -> dict | None:
    paths = scenario_checkpoint_paths(scenario_name)
    if not paths["done"].exists():
        return None
    metrics = read_checkpoint_csv(paths["metrics"])
    if scenario_name == "transfer_malaysia":
        has_classification = any(row.get("task") == "classification" for row in metrics)
        if not has_classification:
            return None
    return {
        "metrics": metrics,
        "skipped": read_checkpoint_csv(paths["skipped"]),
        "predictions": read_checkpoint_csv(paths["predictions"]),
        "future": read_checkpoint_csv(paths["future"]),
        "models": read_checkpoint_csv(paths["models"]),
    }


def write_scenario_checkpoint(
    scenario_name: str,
    metrics: list[dict],
    skipped: list[dict],
    predictions: list[dict],
    future: list[dict],
    models: list[dict],
) -> None:
    paths = scenario_checkpoint_paths(scenario_name)
    pd.DataFrame(metrics).to_csv(paths["metrics"], index=False)
    pd.DataFrame(skipped).to_csv(paths["skipped"], index=False)
    pd.DataFrame(predictions).to_csv(paths["predictions"], index=False)
    pd.DataFrame(future).to_csv(paths["future"], index=False)
    pd.DataFrame(models).to_csv(paths["models"], index=False)
    paths["done"].write_text(json.dumps({"scenario": scenario_name, "status": "complete"}, indent=2), encoding="utf-8")


def model_path(scenario: str, task: str, target: str, model_name: str) -> Path:
    safe_target = target.replace("/", "_")
    return MODELS_DIR / f"{scenario}_{task}_{safe_target}_{model_name}.joblib"


def save_model_artifacts(model, scenario, task, target, model_name, info, cv_rows) -> None:
    path = model_path(scenario, task, target, model_name)
    joblib.dump(model, path)
    write_json(path.with_suffix(".json"), info)
    pd.DataFrame(cv_rows).to_csv(path.with_suffix(".cv_results.csv"), index=False)


def save_iterative_regression_loss(model, X_train, y_train, X_val, y_val, scenario, target, model_name) -> None:
    if model_name not in {"xgboost", "lightgbm"}:
        return
    pre = model.named_steps["preprocessor"]
    est = model.named_steps["model"]
    Xtr = pre.transform(X_train)
    Xva = pre.transform(X_val)
    params = est.get_params()
    params["n_estimators"] = int(params.get("n_estimators", 150))
    fresh = est.__class__(**params)
    try:
        fresh.fit(Xtr, y_train, eval_set=[(Xtr, y_train), (Xva, y_val)], verbose=False)
        history = extract_eval_history(fresh, model_name)
        if not history.empty:
            out = PLOTS_DIR / "loss_curves" / f"{scenario}_{target}_{model_name}_loss.csv"
            history.to_csv(out, index=False)
            save_loss_curve(history, out.with_suffix(".png"), f"{scenario} {target} {model_name}")
    except Exception:
        return


def save_iterative_classification_loss(model, X_train, y_train, X_val, y_val, scenario, target, model_name) -> None:
    pre = model.named_steps["preprocessor"]
    est = model.named_steps["model"]
    Xtr = pre.transform(X_train)
    Xva = pre.transform(X_val)
    params = est.get_params()
    params["n_estimators"] = int(params.get("n_estimators", 150))
    fresh = est.__class__(**params)
    try:
        fresh.fit(Xtr, y_train, eval_set=[(Xtr, y_train), (Xva, y_val)], verbose=False)
        history = extract_eval_history(fresh, model_name)
        if not history.empty:
            out = PLOTS_DIR / "loss_curves" / f"{scenario}_{target}_{model_name}_loss.csv"
            history.to_csv(out, index=False)
            save_loss_curve(history, out.with_suffix(".png"), f"{scenario} {target} {model_name}")
    except Exception:
        return


def extract_eval_history(est, model_name: str) -> pd.DataFrame:
    rows = []
    if model_name == "xgboost" and hasattr(est, "evals_result"):
        result = est.evals_result()
        train_key, valid_key = list(result.keys())[:2]
        metric = list(result[train_key].keys())[0]
        for i, (train, valid) in enumerate(zip(result[train_key][metric], result[valid_key][metric]), start=1):
            rows.append({"iteration": i, "train": train, "valid": valid, "metric": metric})
    elif model_name == "lightgbm" and hasattr(est, "evals_result_"):
        result = est.evals_result_
        train_key, valid_key = list(result.keys())[:2]
        metric = list(result[train_key].keys())[0]
        for i, (train, valid) in enumerate(zip(result[train_key][metric], result[valid_key][metric]), start=1):
            rows.append({"iteration": i, "train": train, "valid": valid, "metric": metric})
    return pd.DataFrame(rows)


def write_dependency_report(skipped_rows: list[dict]) -> None:
    deps = {
        "xgboost": optional_module("xgboost"),
        "lightgbm": optional_module("lightgbm"),
        "statsmodels_sarimax": optional_module("statsmodels"),
        "prophet": optional_module("prophet"),
        "torch": optional_module("torch"),
        "tensorflow": optional_module("tensorflow"),
        "sarimax_baseline": "enabled" if optional_module("statsmodels") else "skipped because statsmodels is not installed",
        "prophet_baseline": "enabled" if optional_module("prophet") else "skipped because prophet is not installed",
        "neural_models": "enabled with PyTorch LSTM, GRU, and temporal CNN" if torch_available() else "skipped because torch is not installed",
    }
    (REPORTS_DIR / "dependency_status.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
