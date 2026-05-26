import warnings

import numpy as np
import pandas as pd

from .evaluation import regression_metrics, add_interval_metrics


def fit_sarimax_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, float]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        raise RuntimeError(f"statsmodels SARIMAX unavailable: {exc}") from exc

    signal_col = target_col.replace("target_next_", "")
    preds = []
    residuals = []
    for country, test_part in test_df.groupby("COUNTRY_CODE"):
        hist_part = train_df[train_df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        test_part = test_part.sort_values("ISO_WEEKSTARTDATE")
        history = hist_part[signal_col].astype(float).dropna().to_list()
        if len(history) < 30:
            preds.extend([np.nan] * len(test_part))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    history,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 0, 52) if len(history) >= 120 else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=75)
                country_preds = np.asarray(fit.forecast(len(test_part)), dtype=float)
                residuals.extend(np.asarray(fit.resid, dtype=float)[-50:])
        except Exception:
            country_preds = np.repeat(float(np.mean(history[-4:])), len(test_part))
        preds.extend(np.maximum(country_preds, 0.0).tolist())
    residual_std = float(np.nanstd(residuals)) if residuals else np.nan
    return np.asarray(preds), residual_std


def fit_prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, float]:
    try:
        from prophet import Prophet
    except Exception as exc:
        raise RuntimeError(f"Prophet unavailable: {exc}") from exc

    signal_col = target_col.replace("target_next_", "")
    preds = []
    residuals = []
    for country, test_part in test_df.groupby("COUNTRY_CODE"):
        hist_part = train_df[train_df["COUNTRY_CODE"].eq(country)].sort_values("ISO_WEEKSTARTDATE")
        test_part = test_part.sort_values("ISO_WEEKSTARTDATE")
        history = hist_part[["ISO_WEEKSTARTDATE", signal_col]].dropna().rename(columns={"ISO_WEEKSTARTDATE": "ds", signal_col: "y"})
        if len(history) < 40:
            preds.extend([np.nan] * len(test_part))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
                model.fit(history)
                future = pd.DataFrame({"ds": test_part["ISO_WEEKSTARTDATE"].to_numpy() + np.timedelta64(7, "D")})
                forecast = model.predict(future)
                country_preds = forecast["yhat"].to_numpy(dtype=float)
                fitted = model.predict(history[["ds"]])["yhat"].to_numpy()
                residuals.extend((history["y"].to_numpy() - fitted)[-50:])
        except Exception:
            country_preds = np.repeat(float(history["y"].tail(4).mean()), len(test_part))
        preds.extend(np.maximum(country_preds, 0.0).tolist())
    residual_std = float(np.nanstd(residuals)) if residuals else np.nan
    return np.asarray(preds), residual_std


def statistical_metric_row(train_df, test_df, target_col, model_name):
    if model_name == "sarimax":
        pred, residual_std = fit_sarimax_forecast(train_df, test_df, target_col)
    elif model_name == "prophet":
        pred, residual_std = fit_prophet_forecast(train_df, test_df, target_col)
    else:
        raise ValueError(model_name)
    metrics = add_interval_metrics(regression_metrics(test_df[target_col], pred), test_df[target_col], pred, residual_std)
    return pred, residual_std, metrics
