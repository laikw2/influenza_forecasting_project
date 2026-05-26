import numpy as np
import pandas as pd

from .config import CORE_SIGNAL_COLUMNS


LAGS = [1, 2, 3, 4, 8, 12]
WINDOWS = [2, 4, 8, 12]


def _rolling_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(y))[mask]
    y = y[mask]
    return float(np.polyfit(x, y, 1)[0])


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["COUNTRY_CODE", "ISO_WEEKSTARTDATE"]).copy()
    iso_week = df["ISO_WEEK"].astype(float)
    df["week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    df["week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    df["month"] = df["ISO_WEEKSTARTDATE"].dt.month
    df["quarter"] = df["ISO_WEEKSTARTDATE"].dt.quarter

    for col in CORE_SIGNAL_COLUMNS:
        if col not in df.columns:
            continue
        df[f"{col}_missing"] = df[col].isna().astype(int)
        grouped = df.groupby("COUNTRY_CODE", group_keys=False)[col]
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = grouped.shift(lag)
        for window in WINDOWS:
            shifted = grouped.shift(1)
            df[f"{col}_roll{window}_mean"] = shifted.groupby(df["COUNTRY_CODE"]).rolling(window, min_periods=2).mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{window}_std"] = shifted.groupby(df["COUNTRY_CODE"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"{col}_roll{window}_slope"] = shifted.groupby(df["COUNTRY_CODE"]).rolling(window, min_periods=2).apply(_rolling_slope, raw=True).reset_index(level=0, drop=True)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["COUNTRY_CODE", "ISO_WEEKSTARTDATE"]).copy()
    group = df.groupby("COUNTRY_CODE", group_keys=False)

    for col in ["INF_A", "INF_B", "INF_ALL", "positivity_rate", "A_rate", "B_rate"]:
        df[f"target_next_{col}"] = group[col].shift(-1)

    current = df["INF_ALL"]
    nxt = df["target_next_INF_ALL"]
    abs_change = nxt - current
    pct_change = abs_change / current.replace(0, np.nan)
    df["target_trend"] = np.select(
        [
            (abs_change >= 5) & (pct_change.fillna(np.inf) >= 0.10),
            (abs_change <= -5) & (pct_change.fillna(-np.inf) <= -0.10),
        ],
        ["increase", "decrease"],
        default="stable",
    )
    df.loc[nxt.isna() | current.isna(), "target_trend"] = np.nan
    df["target_increase_binary"] = np.where(df["target_trend"].eq("increase"), "increase", "not_increase")
    df.loc[df["target_trend"].isna(), "target_increase_binary"] = np.nan

    a_delta = df["target_next_INF_A"] - df["INF_A"]
    b_delta = df["target_next_INF_B"] - df["INF_B"]
    driver = np.full(len(df), "uncertain", dtype=object)
    driver[(a_delta >= 3) & (a_delta > b_delta * 1.25)] = "A"
    driver[(b_delta >= 3) & (b_delta > a_delta * 1.25)] = "B"
    driver[(a_delta >= 3) & (b_delta >= 3) & (np.abs(a_delta - b_delta) <= np.maximum(a_delta, b_delta) * 0.25)] = "both"
    df["target_subtype_driver"] = driver
    df.loc[df[["target_next_INF_A", "target_next_INF_B", "INF_A", "INF_B"]].isna().any(axis=1), "target_subtype_driver"] = np.nan
    return df


def build_modeling_frame(weekly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = add_targets(add_time_series_features(weekly))
    exclude = {
        "COUNTRY_AREA_TERRITORY",
        "WHOREGION",
        "ITZ",
        "HEMISPHERE",
        "origin_sources",
        "ISO_WEEKSTARTDATE",
    }
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude
        and c not in target_cols
        and c != "COUNTRY_CODE"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df, feature_cols
