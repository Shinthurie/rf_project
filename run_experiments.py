"""Reproduce every result for the report in one go.

    python run_experiments.py

Writes CSV tables and PNG figures to ./results
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src import hybrid_ann as HA
from src import hybrid_ts as HT
from src import models as M

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})


def exp1_rf_vs_boosting():
    """Slide: 'Compare RF vs XGBoost on local datasets' - two Sri Lankan classification tasks."""
    tasks = {"rain_tomorrow": ("Rain tomorrow (25 districts, daily 2020-2025)", D.load_rain_tomorrow),
             "dengue_risk": ("High-risk dengue week ahead (25 districts, weekly 2020-2025)", D.load_dengue_risk)}
    for tag, (title, loader) in tasks.items():
        print(f"\n=== Experiment 1 [{tag}]: RF vs XGBoost / LightGBM / CatBoost - {title}, 5-fold CV ===")
        df, t = loader()
        X, y, task = D.prepare_tabular(df, t)
        groups = D.year_groups(df, t)
        print(f"{len(X):,} rows, {X.shape[1]} features, positive rate {y.mean():.1%}")
        # Main result: year-blocked CV (whole years held out - no same-day leakage between districts).
        # Random CV is also run to show how much it flatters the models.
        for kind, g in [("random CV", None), ("year-blocked CV", groups)]:
            res = M.compare_models(X, y, task, groups=g)
            res.insert(0, "CV", kind)
            res.to_csv(OUT / f"exp1_{tag}_{kind.split()[0]}_folds.csv", index=False)
            summ = M.summarise(res.drop(columns="CV"))
            summ.to_csv(OUT / f"exp1_{tag}_{kind.split()[0]}_summary.csv")
            print(f"--- {kind} ---")
            print(summ.to_string())
        cv = res  # plot the year-blocked result

        plot_exp1(cv, title, OUT / f"exp1_{tag}.png")


def plot_exp1(cv, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m in zip(axes, ["Accuracy", "F1", "ROC-AUC"]):
        g = cv.groupby("Model")[m]
        mean, std = g.mean().reindex(M.ALL_MODELS), g.std().reindex(M.ALL_MODELS)
        ax.bar(mean.index, mean, yerr=std, capsize=4, color=["#2E7D32", "#1565C0", "#F9A825", "#AD1457"])
        ax.set_title(m)
        ax.set_ylim(max(0, (mean - std).min() - 0.05), min(1.0, (mean + std).max() + 0.05))
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"{title} - year-blocked CV (mean ± std)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def exp2_rf_arima():
    print("\n=== Experiment 2: RF + ARIMA hybrid (Sri Lanka weekly dengue, last 104 weeks held out) ===")
    rows = []
    for district in ["Colombo", "Gampaha", "Kalutara", "Kandy", "Kurunegala", "Jaffna", "All island"]:
        s = D.load_dengue_series(district, "2010-01-01")
        r = HT.run_hybrid(s, test_size=104)
        for model, met in r["metrics"].iterrows():
            rows.append({"District": district, "ARIMA order": str(r["order"]), "Model": model, **met.to_dict()})
        print(f"{district:12s} order={r['order']}  best={r['metrics']['RMSE'].idxmin()}")
        if district == "Colombo":
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(r["test_dates"], r["actual"], color="#212121", lw=1.6, label="Actual")
            for k, c in [("ARIMA", "#1565C0"), ("Hybrid A: ARIMA + RF(residuals)", "#AD1457"),
                         ("Hybrid B: RF(with ARIMA feature)", "#F57C00")]:
                ax.plot(r["test_dates"], r["preds"][k], color=c, lw=1.1, label=k)
            ax.set_title("Colombo weekly dengue cases - one-step-ahead forecasts on the 104-week test period")
            ax.set_ylabel("cases / week")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(OUT / "exp2_colombo_forecasts.png")
            plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exp2_rf_arima_by_district.csv", index=False)
    pivot = res.pivot(index="District", columns="Model", values="RMSE")
    pivot.to_csv(OUT / "exp2_rmse_pivot.csv")
    print(pivot.round(1).to_string())
    gain = 100 * (1 - pivot.div(pivot["ARIMA"], axis=0))
    print("\nRMSE improvement over ARIMA (%):\n", gain.drop(columns="ARIMA").round(1).to_string())


def exp3_rf_ann():
    print("\n=== Experiment 3: RF + ANN hybrids - next week's dengue cases from lagged weather (year-blocked CV) ===")
    df, t = D.load_dengue_cases()
    X, y, _ = D.prepare_tabular(df, t)
    print(f"{len(X):,} district-weeks, {X.shape[1]} features; target = log(1 + cases next week)")
    folds, summ, oof = HA.evaluate(X, y, groups=D.year_groups(df, t))   # year-blocked CV
    # translate back to case counts for interpretation
    cases = np.expm1(y)
    summ[("MAE (cases)", "")] = [np.mean(np.abs(np.expm1(oof[m]).clip(0) - cases)) for m in summ.index]
    folds.to_csv(OUT / "exp3_rf_ann_folds.csv", index=False)
    summ.to_csv(OUT / "exp3_rf_ann_summary.csv")
    print(summ.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    order = list(summ.index)
    ax.boxplot([folds.loc[folds.Model == m, "RMSE"] for m in order], showmeans=True)
    ax.set_xticks(range(1, len(order) + 1),
                  [m.split(":")[0].replace("Random Forest", "RF").replace("ANN (MLP)", "ANN") for m in order])
    ax.set_ylabel("RMSE on log(1+cases) - lower is better")
    ax.set_title("RF + ANN hybrids - next-week dengue cases, 25 Sri Lankan districts, year-blocked CV")
    fig.tight_layout()
    fig.savefig(OUT / "exp3_rf_ann.png")
    plt.close(fig)

    # which weather lags matter? (RF importance on the full data)
    rf = HA.make_rf().fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(OUT / "exp3_feature_importance.csv", header=["importance"])
    print("\nTop features:\n", imp.head(12).round(4).to_string())


if __name__ == "__main__":
    exp1_rf_vs_boosting()
    exp2_rf_arima()
    exp3_rf_ann()
    print(f"\nAll results saved to {OUT.resolve()}")
