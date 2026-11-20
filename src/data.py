"""Dataset loaders and preprocessing - all data is Sri Lankan (local).

Built by build_datasets.py from two raw sources in data/raw/:
  * Weekly dengue cases per district (Epidemiology Unit / NDCU, via denguedatahub)
  * Daily district weather 2020-2025 (temperature, humidity, rainfall, wind)

Tabular tasks
  * Rain tomorrow?               (classification, 50,967 district-days)
  * High-risk dengue week ahead? (classification, 7,051 district-weeks)
  * Dengue cases next week       (regression,     7,051 district-weeks)
Time series
  * Weekly dengue cases per district / all island (2006 to 2026)
  * Weekly rainfall per district (2020 to 2025)
Any CSV can also be used through `prepare_tabular`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

TABULAR = {
    "Rain tomorrow - 25 districts (classification)": "rain",
    "High-risk dengue week ahead (classification)": "dengue_risk",
    "Dengue cases next week (regression)": "dengue_cases",
}


# --------------------------------------------------------------------------- #
# Tabular datasets
# --------------------------------------------------------------------------- #
def load_rain_tomorrow() -> tuple[pd.DataFrame, str]:
    """Will it rain (> 1 mm) tomorrow? Today's and recent weather + season + district."""
    df = pd.read_csv(DATA_DIR / "sl_rain_tomorrow.csv")      # date kept for year-blocked CV
    return df, "RainTomorrow"


def _dengue_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sl_dengue_weather_weekly.csv")  # week_start kept for year-blocked CV


def load_dengue_risk() -> tuple[pd.DataFrame, str]:
    """Will next week's dengue cases be above the district's 75th percentile?"""
    df = _dengue_panel().drop(columns=["cases_next_week", "log_cases_next_week"])
    return df, "high_risk_next_week"


def load_dengue_cases() -> tuple[pd.DataFrame, str]:
    """log(1 + dengue cases next week) from lagged cases and lagged weather."""
    df = _dengue_panel().drop(columns=["cases_next_week", "high_risk_next_week"])
    return df, "log_cases_next_week"


def load_tabular(label: str) -> tuple[pd.DataFrame, str]:
    return {"rain": load_rain_tomorrow, "dengue_risk": load_dengue_risk,
            "dengue_cases": load_dengue_cases}[TABULAR[label]]()


def _looks_like_date(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty or not sample.str.contains(r"\d").all():
        return False
    return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9


def prepare_tabular(df: pd.DataFrame, target: str):
    """One-hot encode categoricals, fill missing values, return X, y, task."""
    df = df.dropna(subset=[target]).copy()
    y = df.pop(target)

    for c in list(df.columns):
        if df[c].dtype == object or str(df[c].dtype) in ("string", "str"):
            # drop ID-like columns (unique per row) and date columns
            if df[c].nunique() > 0.9 * len(df) or _looks_like_date(df[c]):
                df = df.drop(columns=c)

    cat_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns
    num_cols = df.columns.difference(cat_cols)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    X = pd.get_dummies(df, columns=list(cat_cols), drop_first=True, dtype=float)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]

    task = infer_task(y)
    if task == "classification" and not np.issubdtype(y.dtype, np.number):
        y = pd.Series(pd.factorize(y)[0], index=y.index, name=target)
    return X, y.astype(float if task == "regression" else int), task


def year_groups(df: pd.DataFrame, target: str | None = None):
    """Calendar year of each row (from the first date-like column), for year-blocked CV.
    Returns None if the table has no date column."""
    for c in df.columns:
        if c != target and (df[c].dtype == object or str(df[c].dtype) in ("string", "str")) \
                and _looks_like_date(df[c]):
            yrs = pd.to_datetime(df[c], errors="coerce", format="mixed").dt.year
            if yrs.notna().all() and yrs.nunique() >= 3:
                return yrs.to_numpy()
    return None


def infer_task(y: pd.Series) -> str:
    if not np.issubdtype(y.dtype, np.number) or y.nunique() <= 10:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def load_dengue_long() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "srilanka_dengue_weekly.csv")
    df["date"] = pd.to_datetime(df["start.date"], format="mixed")
    return df[["date", "year", "week", "district", "cases"]]


def dengue_districts() -> list[str]:
    return sorted(load_dengue_long()["district"].unique())


def load_dengue_series(district: str = "Colombo", start: str | None = "2010-01-01") -> pd.Series:
    """Weekly dengue cases for one district (or 'All island')."""
    df = load_dengue_long()
    if district != "All island":
        df = df[df["district"] == district]
    s = df.groupby("date")["cases"].sum().sort_index()
    if start:
        s = s[s.index >= start]
    # Records are consecutive epidemiological weeks (start weekday shifts over the
    # years, so we keep the real dates rather than forcing a calendar grid).
    s = s.astype(float)
    s.name = f"Dengue cases - {district}"
    return s


def weather_districts() -> list[str]:
    return sorted(pd.read_csv(RAW_DIR / "srilanka_weather_daily_2020_2025.csv", usecols=["district"])
                  ["district"].unique())


def load_rain_series(district: str = "Colombo") -> pd.Series:
    """Weekly total rainfall (mm) for one district, 2020-2025."""
    w = pd.read_csv(RAW_DIR / "srilanka_weather_daily_2020_2025.csv", usecols=["date", "district", "precip"])
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"])
    w = w[w["district"] == district].groupby("date")["precip"].mean()
    s = w.resample("W-SUN").sum(min_count=5).dropna()
    s.name = f"Weekly rainfall (mm) - {district}"
    return s
