import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    NUMERIC_COLUMNS,
    RAW_FLUNET_CSV,
    RAW_METADATA_CSV,
    TARGET_COUNTRIES,
    TRANSFER_POOLS,
)


BASE_COLUMNS = [
    "WHOREGION",
    "FLUSEASON",
    "HEMISPHERE",
    "ITZ",
    "COUNTRY_CODE",
    "COUNTRY_AREA_TERRITORY",
    "ISO_WEEKSTARTDATE",
    "ISO_YEAR",
    "ISO_WEEK",
    "ORIGIN_SOURCE",
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_flunet(country_codes: list[str] | None = None) -> pd.DataFrame:
    usecols = BASE_COLUMNS + NUMERIC_COLUMNS
    df = pd.read_csv(RAW_FLUNET_CSV, usecols=lambda col: col in usecols, low_memory=False)
    df["ISO_WEEKSTARTDATE"] = pd.to_datetime(df["ISO_WEEKSTARTDATE"])
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if country_codes:
        df = df[df["COUNTRY_CODE"].isin(country_codes)].copy()
    return df


def load_metadata() -> pd.DataFrame:
    if RAW_METADATA_CSV.exists():
        return pd.read_csv(RAW_METADATA_CSV)
    return pd.DataFrame()


def make_country_week_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["origin_norm"] = raw["ORIGIN_SOURCE"].fillna("NOTDEFINED").str.upper()
    raw["has_sentinel_row"] = raw["origin_norm"].eq("SENTINEL").astype(int)
    raw["has_nonsentinel_row"] = raw["origin_norm"].eq("NONSENTINEL").astype(int)
    raw["has_notdefined_row"] = raw["origin_norm"].eq("NOTDEFINED").astype(int)

    group_cols = [
        "COUNTRY_CODE",
        "COUNTRY_AREA_TERRITORY",
        "WHOREGION",
        "ITZ",
        "HEMISPHERE",
        "ISO_WEEKSTARTDATE",
        "ISO_YEAR",
        "ISO_WEEK",
    ]
    agg_spec = {col: (col, lambda s: s.sum(min_count=1)) for col in NUMERIC_COLUMNS if col in raw.columns}
    agg_spec.update(
        {
            "source_row_count": ("ORIGIN_SOURCE", "size"),
            "has_sentinel": ("has_sentinel_row", "max"),
            "has_nonsentinel": ("has_nonsentinel_row", "max"),
            "has_notdefined": ("has_notdefined_row", "max"),
            "origin_sources": ("origin_norm", lambda s: "|".join(sorted(set(s.dropna())))),
        }
    )
    weekly = raw.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    weekly["is_reported_week"] = 1
    weekly = add_rates(weekly)
    weekly = add_reporting_eras(weekly)
    return weekly.sort_values(["COUNTRY_CODE", "ISO_WEEKSTARTDATE"]).reset_index(drop=True)


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    denom = df["SPEC_PROCESSED_NB"].where(df["SPEC_PROCESSED_NB"] > 0)
    df["positivity_rate"] = df["INF_ALL"] / denom
    df["A_rate"] = df["INF_A"] / denom
    df["B_rate"] = df["INF_B"] / denom
    df["log_spec_processed"] = np.log1p(df["SPEC_PROCESSED_NB"])
    return df


def add_reporting_eras(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["era_2020plus"] = (df["ISO_WEEKSTARTDATE"] >= pd.Timestamp("2020-01-01")).astype(int)
    df["era_2023plus"] = (df["ISO_WEEKSTARTDATE"] >= pd.Timestamp("2023-01-01")).astype(int)
    df["era_2024plus"] = (df["ISO_WEEKSTARTDATE"] >= pd.Timestamp("2024-01-01")).astype(int)
    return df


def reindex_weekly_calendar(weekly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for country, part in weekly.groupby("COUNTRY_CODE"):
        part = part.sort_values("ISO_WEEKSTARTDATE")
        full_dates = pd.date_range(part["ISO_WEEKSTARTDATE"].min(), part["ISO_WEEKSTARTDATE"].max(), freq="W-MON")
        meta = part[
            ["COUNTRY_CODE", "COUNTRY_AREA_TERRITORY", "WHOREGION", "ITZ", "HEMISPHERE"]
        ].drop_duplicates().iloc[0].to_dict()
        full = pd.DataFrame({"ISO_WEEKSTARTDATE": full_dates})
        for key, value in meta.items():
            full[key] = value
        merged = full.merge(part, on=["COUNTRY_CODE", "COUNTRY_AREA_TERRITORY", "WHOREGION", "ITZ", "HEMISPHERE", "ISO_WEEKSTARTDATE"], how="left")
        merged["is_reported_week"] = merged["is_reported_week"].fillna(0).astype(int)
        iso = merged["ISO_WEEKSTARTDATE"].dt.isocalendar()
        merged["ISO_YEAR"] = merged["ISO_YEAR"].fillna(iso.year).astype(int)
        merged["ISO_WEEK"] = merged["ISO_WEEK"].fillna(iso.week).astype(int)
        for flag in ["source_row_count", "has_sentinel", "has_nonsentinel", "has_notdefined"]:
            merged[flag] = merged[flag].fillna(0).astype(int)
        merged["origin_sources"] = merged["origin_sources"].fillna("MISSING_REPORT")
        merged = add_rates(merged)
        merged = add_reporting_eras(merged)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["COUNTRY_CODE", "ISO_WEEKSTARTDATE"])


def prepare_core_datasets() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    all_codes = sorted(set(TARGET_COUNTRIES) | {c for pool in TRANSFER_POOLS.values() for c in pool})
    target_raw = load_flunet(list(TARGET_COUNTRIES))
    transfer_raw = load_flunet(all_codes)

    target_weekly = reindex_weekly_calendar(make_country_week_table(target_raw))
    transfer_weekly = reindex_weekly_calendar(make_country_week_table(transfer_raw))

    target_weekly.to_csv(DATA_DIR / "target_country_weekly_clean.csv", index=False)
    transfer_weekly.to_csv(DATA_DIR / "transfer_country_weekly_clean.csv", index=False)

    audit = build_data_audit(target_raw, target_weekly, transfer_weekly)
    (DATA_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    metadata = load_metadata()
    if not metadata.empty:
        metadata.to_csv(DATA_DIR / "flunet_metadata_copy.csv", index=False)

    return {
        "target_weekly": target_weekly,
        "transfer_weekly": transfer_weekly,
        "audit": audit,
    }


def build_data_audit(raw: pd.DataFrame, target_weekly: pd.DataFrame, transfer_weekly: pd.DataFrame) -> dict:
    audit = {
        "raw_file": str(RAW_FLUNET_CSV),
        "raw_rows_for_target_countries": int(len(raw)),
        "target_countries": TARGET_COUNTRIES,
        "note": "FluNet values are laboratory surveillance reports, not national incidence counts.",
        "countries": {},
        "transfer_country_count": int(transfer_weekly["COUNTRY_CODE"].nunique()),
    }
    for country, part in target_weekly.groupby("COUNTRY_CODE"):
        reported = part[part["is_reported_week"].eq(1)]
        audit["countries"][country] = {
            "country_name": TARGET_COUNTRIES.get(country, country),
            "first_week": str(part["ISO_WEEKSTARTDATE"].min().date()),
            "last_week": str(part["ISO_WEEKSTARTDATE"].max().date()),
            "calendar_weeks": int(len(part)),
            "reported_weeks": int(reported["ISO_WEEKSTARTDATE"].nunique()),
            "missing_calendar_weeks": int(len(part) - reported["ISO_WEEKSTARTDATE"].nunique()),
            "source_rows_by_type": raw[raw["COUNTRY_CODE"].eq(country)]["ORIGIN_SOURCE"].fillna("NOTDEFINED").value_counts().to_dict(),
        }
    return audit
