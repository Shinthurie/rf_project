"""Random Forest and gradient-boosting models + a fair cross-validated comparison."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

SEED = 42


def make_model(name: str, task: str, **params):
    """Return an unfitted model. Defaults are sensible, untuned baselines so the
    comparison is fair (every model gets the same budget: 300 trees)."""
    clf = task == "classification"
    if name == "Random Forest":
        p = dict(n_estimators=300, n_jobs=-1, random_state=SEED, oob_score=True)
        p.update(params)
        return RandomForestClassifier(**p) if clf else RandomForestRegressor(**p)
    if name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        p = dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                 colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)
        p.update(params)
        return XGBClassifier(eval_metric="logloss", **p) if clf else XGBRegressor(**p)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier, LGBMRegressor
        p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        p.update(params)
        return LGBMClassifier(**p) if clf else LGBMRegressor(**p)
    if name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        p = dict(iterations=300, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0,
                 allow_writing_files=False)
        p.update(params)
        return CatBoostClassifier(**p) if clf else CatBoostRegressor(**p)
    raise ValueError(name)


ALL_MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def score(task, y_true, y_pred, y_prob=None) -> dict:
    if task == "classification":
        out = {"Accuracy": accuracy_score(y_true, y_pred),
               "F1": f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")}
        if y_prob is not None and len(np.unique(y_true)) == 2:
            out["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        return out
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def cv_splits(X, y, task, n_splits=5, groups=None):
    """Random (stratified) k-fold, or - if `groups` (e.g. year) is given - blocked k-fold where
    every fold holds out whole groups, so the same date never appears in train and test."""
    if groups is not None:
        k = min(n_splits, len(np.unique(groups)))
        return list(GroupKFold(k).split(X, y, groups))
    cv = (StratifiedKFold(n_splits, shuffle=True, random_state=SEED) if task == "classification"
          else KFold(n_splits, shuffle=True, random_state=SEED))
    return list(cv.split(X, y))


def compare_models(X: pd.DataFrame, y: pd.Series, task: str, models=ALL_MODELS,
                   n_splits: int = 5, progress=None, groups=None) -> pd.DataFrame:
    """Cross-validation where every model sees identical folds. One row per (model, fold)."""
    splits = cv_splits(X, y, task, n_splits, groups)
    n_splits = len(splits)
    rows, total, done = [], len(models) * n_splits, 0
    for fold, (tr, te) in enumerate(splits):
        for name in models:
            m = make_model(name, task)
            t0 = time.perf_counter()
            m.fit(X.iloc[tr], y.iloc[tr])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = m.predict(X.iloc[te])
            prob = m.predict_proba(X.iloc[te])[:, 1] if task == "classification" and len(np.unique(y)) == 2 else None
            pred_s = time.perf_counter() - t0
            r = {"Model": name, "Fold": fold + 1, **score(task, y.iloc[te], np.ravel(pred), prob),
                 "Train time (s)": fit_s, "Predict time (s)": pred_s}
            rows.append(r)
            done += 1
            if progress:
                progress(done / total, f"{name} - fold {fold + 1}/{n_splits}")
    return pd.DataFrame(rows)


def summarise(cv_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in cv_df.columns if c not in ("Model", "Fold")]
    agg = cv_df.groupby("Model")[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = [f"{a:.4f} ± {b:.4f}" if "time" not in m else f"{a:.2f}"
                  for a, b in zip(agg[(m, "mean")], agg[(m, "std")])]
    return out


def oob_curve(X, y, task, max_trees=300, step=20, **params) -> pd.DataFrame:
    """OOB error as the forest grows - shows why more trees stop helping."""
    base = dict(warm_start=True, oob_score=True, n_jobs=-1, random_state=SEED)
    base.update(params)
    m = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(n_estimators=step, **base)
    rows = []
    for n in range(step, max_trees + 1, step):
        m.set_params(n_estimators=n)
        m.fit(X, y)
        err = 1 - m.oob_score_  # 1-accuracy (clf) or 1-R2 (reg)
        rows.append({"Trees": n, "OOB error": err})
    return pd.DataFrame(rows)
