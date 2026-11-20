"""Build the local (Sri Lankan) modelling tables from the two raw sources.

    python build_datasets.py

Raw inputs (data/raw/):
  * srilanka_dengue_weekly.csv          - weekly dengue cases per district, 2006 to Sep 2026
                                          (Epidemiology Unit / NDCU, via the denguedatahub package)
  * srilanka_weather_daily_2020_2025.csv - daily temperature, humidity, rainfall and wind speed
                                          for 25 districts, Jan 2020 to Sep 2025

Outputs (data/):
  * sl_rain_tomorrow.csv          - classification: will it rain (> 1 mm) tomorrow in this district?
  * sl_dengue_weather_weekly.csv  - weekly district panel: dengue cases + weather, with lags
        targets: high_risk_next_week (classification), log_cases_next_week (regression)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data"
RAIN_THRESHOLD_MM = 1.0
NAME_FIX = {"Hambanthota": "Hambantota", "NuwaraEliya": "Nuwara Eliya"}


def clean_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / "srilanka_weather_daily_2020_2025.csv")
    n0 = len(w)
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"])
    # a few district-days appear twice -> average them
    w = (w.groupby(["district", "date"], as_index=False)[["temp", "humidity", "precip", "windspeed"]].mean())
    print(f"weather: {n0:,} raw rows -> {len(w):,} clean district-days "
          f"({w.district.nunique()} districts, {w.date.min().date()} to {w.date.max().date()})")
    return w


def build_rain_tomorrow(w: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for d, g in w.groupby("district"):
        g = g.set_index("date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))  # gaps stay NaN
        g["district"] = d
        f = pd.DataFrame(index=g.index)
        f["district"] = d
        for c in ["temp", "humidity", "precip", "windspeed"]:
            f[c] = g[c]
            f[f"{c}_lag1"] = g[c].shift(1)
            f[f"{c}_lag2"] = g[c].shift(2)
        f["humidity_change"] = g["humidity"] - g["humidity"].shift(1)
        f["temp_change"] = g["temp"] - g["temp"].shift(1)
        f["rain_3day_total"] = g["precip"].rolling(3).sum()
        f["rain_7day_total"] = g["precip"].rolling(7).sum()
        f["rainy_days_last7"] = (g["precip"] > RAIN_THRESHOLD_MM).rolling(7).sum()
        doy = f.index.dayofyear
        f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        f["month"] = f.index.month
        nxt = g["precip"].shift(-1)
        f["RainTomorrow"] = np.where(nxt.isna(), np.nan, (nxt > RAIN_THRESHOLD_MM).astype(float))
        parts.append(f)
    df = pd.concat(parts).dropna().reset_index(names="date")
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)
    print(f"rain-tomorrow table: {len(df):,} rows, rain-tomorrow rate {df.RainTomorrow.mean():.1%}")
    return df


def weekly_weather(w: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather onto the dengue surveillance weeks of each district."""
    w = w.sort_values("date")
    wk = weeks[["district", "week_start"]].drop_duplicates().sort_values("week_start")
    m = pd.merge_asof(w, wk, left_on="date", right_on="week_start", by="district", direction="backward")
    m = m[(m["date"] - m["week_start"]).dt.days.between(0, 6)]
    agg = m.groupby(["district", "week_start"]).agg(
        temp_mean=("temp", "mean"), humidity_mean=("humidity", "mean"),
        rain_total=("precip", "sum"), rainy_days=("precip", lambda s: (s > RAIN_THRESHOLD_MM).sum()),
        wind_mean=("windspeed", "mean"), n_days=("temp", "size")).reset_index()
    return agg[agg["n_days"] >= 5].drop(columns="n_days")


def build_dengue_weather(w: pd.DataFrame) -> pd.DataFrame:
    dg = pd.read_csv(RAW / "srilanka_dengue_weekly.csv")
    dg["week_start"] = pd.to_datetime(dg["start.date"], format="mixed")
    dg["district"] = dg["district"].replace(NAME_FIX)
    dg = dg[dg["district"].isin(w["district"].unique())]          # Kalmunai has no weather data
    dg = dg.groupby(["district", "week_start"], as_index=False)["cases"].sum()

    ww = weekly_weather(w, dg)
    panel = dg.merge(ww, on=["district", "week_start"], how="inner").sort_values(["district", "week_start"])

    parts = []
    for d, g in panel.groupby("district"):
        g = g.reset_index(drop=True)
        f = pd.DataFrame({"week_start": g["week_start"], "district": d})
        lc = np.log1p(g["cases"])
        for k in range(0, 4):
            f[f"log_cases_lag{k}"] = lc.shift(k)
        f["cases_trend"] = lc - lc.shift(1)
        # mosquitoes breed after rain: effect on cases appears with a lag of ~1-2 months
        for k in [0, 1, 2, 3, 4, 6, 8]:
            f[f"rain_total_lag{k}"] = g["rain_total"].shift(k)
        for k in [0, 2, 4, 6]:
            f[f"humidity_lag{k}"] = g["humidity_mean"].shift(k)
            f[f"temp_lag{k}"] = g["temp_mean"].shift(k)
        f["rain_8wk_total"] = g["rain_total"].rolling(8).sum()
        f["rainy_days_lag0"] = g["rainy_days"]
        f["wind_lag0"] = g["wind_mean"]
        woy = g["week_start"].dt.isocalendar().week.astype(float)
        f["season_sin"] = np.sin(2 * np.pi * woy / 52.18)
        f["season_cos"] = np.cos(2 * np.pi * woy / 52.18)
        nxt = g["cases"].shift(-1)
        # consecutive weeks only (weather gaps would otherwise join non-adjacent weeks)
        gap = g["week_start"].diff().dt.days.fillna(7)
        ok = gap.rolling(9, min_periods=1).max() <= 8
        f["cases_next_week"] = nxt
        f = f[ok.values]
        parts.append(f)
    df = pd.concat(parts).dropna().reset_index(drop=True)

    # "high risk" = next week's cases above the district's own 75th percentile
    p75 = df.groupby("district")["cases_next_week"].transform(lambda s: s.quantile(0.75))
    df["high_risk_next_week"] = (df["cases_next_week"] > p75).astype(int)
    df["log_cases_next_week"] = np.log1p(df["cases_next_week"])
    print(f"dengue-weather panel: {len(df):,} district-weeks, {df.district.nunique()} districts, "
          f"{df.week_start.min().date()} to {df.week_start.max().date()}, "
          f"high-risk rate {df.high_risk_next_week.mean():.1%}")
    return df


if __name__ == "__main__":
    w = clean_weather()
    build_rain_tomorrow(w).to_csv(OUT / "sl_rain_tomorrow.csv", index=False)
    build_dengue_weather(w).to_csv(OUT / "sl_dengue_weather_weekly.csv", index=False)
    print("saved to", OUT.resolve())
