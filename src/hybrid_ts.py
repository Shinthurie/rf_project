"""RF + ARIMA hybrid for time series (Zhang-style residual hybrid).

    y_t = L_t + N_t
    ARIMA estimates the linear part L_t; the Random Forest learns the non-linear
    structure N_t left in ARIMA's residuals (using lagged residuals, lagged values and
    seasonality). Final forecast = ARIMA forecast + RF residual forecast.

Two extra models are fitted for comparison:
    * ARIMA alone
    * RF alone (lag features only)
    * Hybrid B: RF that receives the ARIMA forecast as an extra input feature
All forecasts are one-step-ahead on a held-out test period (no look-ahead).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

SEED = 42
warnings.filterwarnings("ignore")


def _fit_arima(y: np.ndarray, order):
    trend = "c" if order[1] == 0 else "n"
    return ARIMA(y, order=order, trend=trend).fit()


def select_arima_order(y: np.ndarray, p_max=3, d_vals=(0, 1), q_max=3):
    best, best_aic = None, np.inf
    for p, d, q in itertools.product(range(p_max + 1), d_vals, range(q_max + 1)):
        if p == q == 0:
            continue
        try:
            aic = _fit_arima(y, (p, d, q)).aic
        except Exception:
            continue
        if aic < best_aic:
            best, best_aic = (p, d, q), aic
    return best


def _features(y: np.ndarray, resid: np.ndarray, dates: pd.DatetimeIndex, n_lags: int,
              arima_pred: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(y)))
    for k in range(1, n_lags + 1):
        df[f"y_lag{k}"] = pd.Series(y).shift(k)
        if resid is not None:
            df[f"resid_lag{k}"] = pd.Series(resid).shift(k)
    woy = dates.isocalendar().week.to_numpy().astype(float)
    df["season_sin"] = np.sin(2 * np.pi * woy / 52.18)
    df["season_cos"] = np.cos(2 * np.pi * woy / 52.18)
    if arima_pred is not None:
        df["arima_pred"] = arima_pred
    return df


def _metrics(actual, pred) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "MAE": float(mean_absolute_error(actual, pred)),
            "sMAPE %": float(100 * np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-9)))}


def run_hybrid(series: pd.Series, test_size: int = 104, order=None, n_lags: int = 8,
               rf_params: dict | None = None, log_transform: bool = True) -> dict:
    dates = pd.DatetimeIndex(series.index)
    raw = series.to_numpy(float)
    y = np.log1p(raw) if log_transform else raw.copy()
    inv = np.expm1 if log_transform else (lambda v: v)
    n = len(y)
    split = n - test_size

    # 1) ARIMA on the training period, then one-step-ahead predictions over the
    #    whole series with the *same* parameters (no refitting on test data).
    if order is None:
        order = select_arima_order(y[:split])
    res_train = _fit_arima(y[:split], order)
    res_full = res_train.apply(y)
    arima_pred = np.asarray(res_full.fittedvalues)
    burn = max(order[1] + 1, n_lags + 1, 5)          # skip unstable first predictions
    resid = y - arima_pred

    rf_kw = dict(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)
    rf_kw.update(rf_params or {})

    # 2) Hybrid A: RF models the ARIMA residuals
    Xa = _features(y, resid, dates, n_lags)
    tr = np.arange(burn, split)
    te = np.arange(split, n)
    rf_resid = RandomForestRegressor(**rf_kw).fit(Xa.iloc[tr], resid[tr])
    hybrid_pred = arima_pred[te] + rf_resid.predict(Xa.iloc[te])

    # 3) RF alone (lags of y + seasonality only)
    Xr = _features(y, None, dates, n_lags)
    rf_only = RandomForestRegressor(**rf_kw).fit(Xr.iloc[tr], y[tr])
    rf_pred = rf_only.predict(Xr.iloc[te])

    # 4) Hybrid B: ARIMA forecast fed to the RF as a feature
    Xb = _features(y, resid, dates, n_lags, arima_pred=arima_pred)
    rf_b = RandomForestRegressor(**rf_kw).fit(Xb.iloc[tr], y[tr])
    hyb_b_pred = rf_b.predict(Xb.iloc[te])

    actual = raw[te]
    preds = {
        "ARIMA": inv(arima_pred[te]),
        "Random Forest": inv(rf_pred),
        "Hybrid A: ARIMA + RF(residuals)": inv(hybrid_pred),
        "Hybrid B: RF(with ARIMA feature)": inv(hyb_b_pred),
    }
    preds = {k: np.clip(v, 0, None) for k, v in preds.items()}
    table = pd.DataFrame({k: _metrics(actual, v) for k, v in preds.items()}).T

    # residual diagnostics: is there non-linear structure left for the RF to learn?
    fi = pd.Series(rf_resid.feature_importances_, index=Xa.columns).sort_values(ascending=False)

    return {
        "order": order, "dates": dates, "actual_all": raw, "split": split, "test_dates": dates[te],
        "actual": actual, "preds": preds, "metrics": table, "arima_aic": res_train.aic,
        "resid_importance": fi, "arima_summary": res_train.summary().as_text(),
        "_state": dict(y=y, resid=resid, res_full=res_full, rf_resid=rf_resid, n_lags=n_lags,
                       inv=inv, dates=dates, log=log_transform),
    }


def forecast_future(result: dict, horizon: int = 12) -> pd.DataFrame:
    """Recursive multi-step forecast of the hybrid beyond the last observation.
    (Models are those fitted on the training window.)"""
    st = result["_state"]
    y, resid = list(st["y"]), list(st["resid"])
    dates = list(st["dates"])
    arima_fc = np.asarray(st["res_full"].forecast(horizon))
    out = []
    for h in range(horizon):
        nd = dates[-1] + pd.Timedelta(days=7)
        dates.append(nd)
        feats = _features(np.array(y + [np.nan]), np.array(resid + [np.nan]),
                          pd.DatetimeIndex(dates), st["n_lags"]).iloc[[-1]]
        r_hat = float(st["rf_resid"].predict(feats)[0])
        y_hat = arima_fc[h] + r_hat
        y.append(y_hat)
        resid.append(r_hat)
        out.append({"week_start": nd, "ARIMA": st["inv"](arima_fc[h]), "Hybrid": st["inv"](y_hat)})
    df = pd.DataFrame(out)
    df[["ARIMA", "Hybrid"]] = df[["ARIMA", "Hybrid"]].clip(lower=0)
    return df
