"""RF + ANN hybrids for tabular regression.

Four ways to combine a Random Forest with an artificial neural network (MLP):
  1. Weighted average   - blend both predictions; weight chosen on a validation split
  2. Stacking           - RF and ANN are base learners, a Ridge meta-model combines
                          their out-of-fold predictions
  3. Residual (RF -> ANN) - RF predicts, the ANN learns what the RF got wrong
                          (trained on the RF's out-of-bag residuals, so no leakage)
  4. RF feature selection -> ANN - the RF's importance ranking picks the inputs the
                          ANN is trained on
Compared against RF alone and ANN alone with k-fold cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

SEED = 42


def make_rf(**kw):
    p = dict(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)
    p.update(kw)
    return RandomForestRegressor(**p)


def make_ann(hidden=(64, 32), alpha=1e-2, max_iter=300):
    mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=2e-3, batch_size=128,
                       max_iter=max_iter, early_stopping=True, validation_fraction=0.15,
                       n_iter_no_change=15, random_state=SEED)
    # scale inputs AND target - MLPs train badly on raw-scale targets
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp),
                                      transformer=StandardScaler())


class WeightedAverage(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(64, 32)):
        self.hidden = hidden

    def fit(self, X, y):
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)
        rf, ann = make_rf().fit(Xtr, ytr), make_ann(self.hidden).fit(Xtr, ytr)
        prf, pann = rf.predict(Xva), ann.predict(Xva)
        ws = np.linspace(0, 1, 21)
        errs = [mean_squared_error(yva, w * prf + (1 - w) * pann) for w in ws]
        self.w_rf_ = float(ws[int(np.argmin(errs))])
        self.rf_, self.ann_ = make_rf().fit(X, y), make_ann(self.hidden).fit(X, y)
        return self

    def predict(self, X):
        return self.w_rf_ * self.rf_.predict(X) + (1 - self.w_rf_) * self.ann_.predict(X)


class ResidualHybrid(BaseEstimator, RegressorMixin):
    """RF first; ANN fitted on the RF's out-of-bag residuals."""
    def __init__(self, hidden=(32, 16)):
        self.hidden = hidden

    def fit(self, X, y):
        self.rf_ = make_rf(oob_score=True).fit(X, y)
        resid = np.asarray(y) - self.rf_.oob_prediction_
        self.ann_ = make_ann(self.hidden, alpha=1e-1).fit(X, resid)
        return self

    def predict(self, X):
        return self.rf_.predict(X) + self.ann_.predict(X)


class RFSelectANN(BaseEstimator, RegressorMixin):
    """Keep the RF's top-k features (by importance), train the ANN on those."""
    def __init__(self, top_k=0.5, hidden=(64, 32)):
        self.top_k, self.hidden = top_k, hidden

    def fit(self, X, y):
        X = pd.DataFrame(X)
        rf = make_rf().fit(X, y)
        k = max(1, int(round(self.top_k * X.shape[1]))) if self.top_k <= 1 else int(self.top_k)
        self.cols_ = list(pd.Series(rf.feature_importances_, index=X.columns).nlargest(k).index)
        self.ann_ = make_ann(self.hidden).fit(X[self.cols_], y)
        return self

    def predict(self, X):
        return self.ann_.predict(pd.DataFrame(X)[self.cols_])


def model_zoo(hidden=(64, 32)) -> dict:
    return {
        "Random Forest": make_rf(),
        "ANN (MLP)": make_ann(hidden),
        "Hybrid 1: Weighted average": WeightedAverage(hidden),
        "Hybrid 2: Stacking (Ridge meta)": StackingRegressor(
            [("rf", make_rf()), ("ann", make_ann(hidden))], final_estimator=RidgeCV(), cv=3, n_jobs=1),
        "Hybrid 3: RF + ANN on residuals": ResidualHybrid(),
        "Hybrid 4: RF-selected features -> ANN": RFSelectANN(hidden=hidden),
    }


def evaluate(X: pd.DataFrame, y: pd.Series, n_splits=5, hidden=(64, 32), progress=None, groups=None):
    from src.models import cv_splits
    zoo = model_zoo(hidden)
    splits = cv_splits(X, y, "regression", n_splits, groups)
    rows, oof = [], {k: np.zeros(len(y)) for k in zoo}
    total, done = len(splits) * len(zoo), 0
    for f, (tr, te) in enumerate(splits):
        for name, est in zoo.items():
            m = clone(est).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            oof[name][te] = p
            rows.append({"Model": name, "Fold": f + 1,
                         "RMSE": float(np.sqrt(mean_squared_error(y.iloc[te], p))),
                         "MAE": mean_absolute_error(y.iloc[te], p),
                         "R2": r2_score(y.iloc[te], p)})
            done += 1
            if progress:
                progress(done / total, f"{name} - fold {f + 1}")
    folds = pd.DataFrame(rows)
    summary = folds.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    summary = summary.reindex(list(zoo))
    return folds, summary, pd.DataFrame(oof, index=y.index)
