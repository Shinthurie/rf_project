"""Random Forest Dashboard - hands-on project.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from src import data as D
from src import hybrid_ann as HA
from src import hybrid_ts as HT
from src import models as M

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Random Forest Dashboard", page_icon="🌲", layout="wide")

COLORS = {"Random Forest": "#2E7D32", "XGBoost": "#1565C0", "LightGBM": "#F9A825", "CatBoost": "#AD1457",
          "ARIMA": "#1565C0", "Hybrid A: ARIMA + RF(residuals)": "#AD1457",
          "Hybrid B: RF(with ARIMA feature)": "#F57C00", "Actual": "#212121"}


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def builtin_dataset(name: str):
    return D.load_tabular(name)


CLASS_LABELS = {"Rain tomorrow": ["No rain (0)", "Rain (1)"],
                "High-risk dengue": ["Normal week (0)", "High risk (1)"]}


def dataset_picker(key: str, allow=tuple(D.TABULAR), max_rows_default=15000):
    choice = st.selectbox("Dataset (Sri Lanka)", list(allow) + ["Upload my own CSV"], key=f"{key}_ds")
    if choice == "Upload my own CSV":
        up = st.file_uploader("CSV file", type="csv", key=f"{key}_up")
        if up is None:
            st.info("Upload a CSV to continue.")
            return None
        df = pd.read_csv(up)
        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1, key=f"{key}_tg")
        name = up.name
    else:
        df, target = builtin_dataset(choice)
        name = choice
    if len(df) > max_rows_default:
        n = st.slider("Rows to use (random sample, for speed)", 2000, len(df),
                      max_rows_default, 1000, key=f"{key}_n")
        if n < len(df):
            df = df.sample(n, random_state=M.SEED)
            name = f"{name} [{n}]"
    return df, target, name


@st.cache_data(show_spinner=False)
def prep(df, target):
    return D.prepare_tabular(df, target)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌲 RF Dashboard")
    st.caption("Hands-on project: Random Forests, boosting comparison and hybrid models")
    st.markdown(
        "**Sections**\n"
        "1. **RF Explorer** - train & inspect a forest\n"
        "2. **RF vs Boosting** - XGBoost, LightGBM, CatBoost\n"
        "3. **RF + ARIMA** - dengue & rainfall time series\n"
        "4. **RF + ANN** - dengue cases regression\n"
        "5. **Research notes**")
    st.divider()
    st.markdown("**🇱🇰 All data is Sri Lankan**")
    st.caption("• Weekly dengue cases, 26 districts, 2006-2026 (Epidemiology Unit / NDCU, via *denguedatahub*)\n\n"
               "• Daily district weather 2020-2025: temperature, humidity, rainfall, wind\n\n"
               "Built into modelling tables by `build_datasets.py`.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🌲 RF Explorer", "⚔️ RF vs Boosting", "📈 RF + ARIMA (time series)", "🧠 RF + ANN (regression)",
     "📚 Research notes"])

# --------------------------------------------------------------------------- #
# 1. RF Explorer
# --------------------------------------------------------------------------- #
def render_rf():
    st.header("Random Forest Explorer")
    st.write("A Random Forest is a **bagged ensemble of decision trees**: each tree is trained on a "
             "bootstrap sample of the rows and considers a random subset of features at each split. "
             "Predictions are averaged (regression) or voted (classification).")
    left, right = st.columns([1, 3])
    with left:
        picked = dataset_picker("rf")
        if picked is None:
            return
        df, target, ds_name = picked
        X, y, task = prep(df, target)
        st.caption(f"{X.shape[0]:,} rows · {X.shape[1]} features · task: **{task}**")
        n_est = st.slider("Number of trees", 10, 500, 200, 10)
        depth = st.slider("Max depth (0 = unlimited)", 0, 30, 0)
        mf_opt = st.selectbox("Max features per split", ["sqrt", "log2", "0.3", "0.5", "1.0 (= bagging)"])
        leaf = st.slider("Min samples per leaf", 1, 50, 1)
        cw = st.checkbox("Balanced class weights", value=False, disabled=task != "classification")
        g1 = D.year_groups(df.dropna(subset=[target]), target)
        time_split = st.checkbox("Test on the latest year (time split)", value=g1 is not None,
                                 disabled=g1 is None,
                                 help="Train on earlier years, test on the most recent one - an honest "
                                      "'predict the future' setup with no same-day leakage.")
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05, disabled=time_split)

    mf = mf_opt if mf_opt in ("sqrt", "log2") else float(mf_opt.split()[0])
    params = dict(n_estimators=n_est, max_depth=depth or None, max_features=mf, min_samples_leaf=leaf,
                  n_jobs=-1, random_state=M.SEED, oob_score=True)
    if task == "classification" and cw:
        params["class_weight"] = "balanced"
    if time_split:
        last = g1.max()
        test_mask = g1 == last
        Xtr, Xte, ytr, yte = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
        split_desc = f"train: {int(g1.min())}-{int(last) - 1} · test: {int(last)}"
    else:
        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=M.SEED, stratify=strat)
        split_desc = f"random {int(test_size * 100)}% test split"
    left.caption(split_desc)

    @st.cache_resource(show_spinner="Training forest...")
    def train_rf(_X, _y, key, **p):
        Model = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return Model(**p).fit(_X, _y)

    rf = train_rf(Xtr, ytr, (ds_name, target, test_size, time_split), **params)
    pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)[:, 1] if task == "classification" and y.nunique() == 2 else None
    sc = M.score(task, yte, pred, prob)

    with right:
        cols = st.columns(len(sc) + 1)
        for c, (k, v) in zip(cols, sc.items()):
            c.metric(k, f"{v:.3f}")
        cols[-1].metric("OOB score", f"{rf.oob_score_:.3f}",
                        help="Out-of-bag: each row is scored only by trees that never saw it - a free validation estimate.")

        a, b = st.columns(2)
        with a:
            st.subheader("Feature importance")
            imp_kind = st.radio("Method", ["Impurity (MDI)", "Permutation (test set)"], horizontal=True)
            if imp_kind.startswith("Impurity"):
                imp = pd.Series(rf.feature_importances_, index=X.columns)
            else:
                with st.spinner("Permuting..."):
                    sub = Xte.sample(min(2000, len(Xte)), random_state=M.SEED)
                    pi = permutation_importance(rf, sub, yte.loc[sub.index], n_repeats=5,
                                                random_state=M.SEED, n_jobs=-1)
                imp = pd.Series(pi.importances_mean, index=X.columns)
            imp = imp.nlargest(15).sort_values()
            st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "importance", "index": ""},
                                   color_discrete_sequence=["#2E7D32"]).update_layout(showlegend=False, height=420),
                            width="stretch")
        with b:
            if task == "classification":
                st.subheader("Confusion matrix")
                cm = confusion_matrix(yte, pred)
                cls = [str(c) for c in sorted(np.unique(np.concatenate([yte, pred])))]
                for prefix, labels in CLASS_LABELS.items():
                    if ds_name.startswith(prefix) and len(cls) == 2:
                        cls = labels
                st.plotly_chart(px.imshow(cm, x=cls, y=cls, text_auto=True, color_continuous_scale="Greens",
                                          labels=dict(x="Predicted", y="Actual", color="count"))
                                .update_layout(height=420).update_xaxes(type="category").update_yaxes(type="category"),
                                width="stretch")
            else:
                st.subheader("Predicted vs actual")
                fig = px.scatter(x=yte, y=pred, labels={"x": "Actual", "y": "Predicted"}, opacity=0.6,
                                 color_discrete_sequence=["#2E7D32"])
                lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
                fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
                st.plotly_chart(fig.update_layout(height=420), width="stretch")

        c, d = st.columns(2)
        with c:
            st.subheader("OOB error vs number of trees")
            curve = M.oob_curve(Xtr, ytr, task, max_trees=max(60, n_est), step=max(10, n_est // 15),
                                max_depth=depth or None, max_features=mf, min_samples_leaf=leaf)
            st.plotly_chart(px.line(curve, x="Trees", y="OOB error", markers=True,
                                    color_discrete_sequence=["#2E7D32"]).update_layout(height=350),
                            width="stretch")
            st.caption("Error falls quickly then flattens: adding trees reduces variance but never "
                       "causes overfitting - it only costs compute.")
        with d:
            if task == "classification" and prob is not None:
                st.subheader("ROC curve")
                fpr, tpr, _ = roc_curve(yte, prob)
                fig = px.area(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"},
                              color_discrete_sequence=["#2E7D32"])
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
                st.plotly_chart(fig.update_layout(height=350), width="stretch")
            else:
                st.subheader("Residuals")
                st.plotly_chart(px.histogram(x=yte - pred, nbins=30, labels={"x": "Actual - predicted"},
                                             color_discrete_sequence=["#2E7D32"]).update_layout(height=350),
                                width="stretch")

        st.subheader("Inside the forest: how individual trees vote")
        e, f = st.columns([2, 3])
        with e:
            # default to the test row the trees disagree on most (most interesting to look at)
            per_tree = np.stack([t.predict(Xte.values) for t in rf.estimators_[:50]])
            default_row = int(np.argmax(per_tree.std(axis=0)))
            i = st.number_input("Test row #", 0, len(Xte) - 1, default_row,
                                help="Defaults to the row the trees disagree on most.")
            row = Xte.iloc[[int(i)]]
            if task == "classification":
                votes = np.array([t.predict_proba(row.values)[0, -1] for t in rf.estimators_])
                xlab = "P(positive class) from each tree"
            else:
                votes = np.array([t.predict(row.values)[0] for t in rf.estimators_])
                xlab = "Prediction from each tree"
            fig = px.histogram(x=votes, nbins=25, labels={"x": xlab}, color_discrete_sequence=["#81C784"])
            fig.add_vline(x=votes.mean(), line_color="#1B5E20", annotation_text="forest average")
            st.plotly_chart(fig.update_layout(height=330), width="stretch")
            st.caption(f"Actual value: **{yte.iloc[int(i)]}** · forest output: **{votes.mean():.3f}** · "
                       f"spread between trees (std): {votes.std():.3f}")
        with f:
            ti = st.slider("Show tree #", 0, len(rf.estimators_) - 1, 0)
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_tree(rf.estimators_[ti], max_depth=2, feature_names=list(X.columns), filled=True,
                      rounded=True, fontsize=8, ax=ax, impurity=False, proportion=True)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Top 3 levels of one tree. Each tree is different because of bootstrap rows and "
                       "random feature subsets - that diversity is what makes the average strong.")

# --------------------------------------------------------------------------- #
# 2. RF vs Boosting
# --------------------------------------------------------------------------- #
def render_cmp():
    st.header("Random Forest vs Gradient Boosting")
    st.write("Every model gets identical cross-validation folds and the same budget of 300 trees "
             "(untuned defaults), so differences come from the algorithm, not from tuning effort.")
    l2, r2 = st.columns([1, 3])
    with l2:
        picked = dataset_picker("cmp")
        if picked is None:
            return
        df2, target2, name2 = picked
        X2, y2, task2 = prep(df2, target2)
        st.caption(f"{X2.shape[0]:,} rows · {X2.shape[1]} features · {task2}")
        chosen = st.multiselect("Models", M.ALL_MODELS, default=M.ALL_MODELS)
        k = st.slider("CV folds", 3, 10, 5)
        g2 = D.year_groups(df2.dropna(subset=[target2]), target2)
        blocked2 = st.checkbox("Year-blocked CV", value=g2 is not None, disabled=g2 is None, key="blk2",
                               help="Each fold holds out whole years, so the same date never appears in both "
                                    "train and test. Random CV leaks information between neighbouring "
                                    "districts on the same day and looks better than it really is.")
        go_btn = st.button("Run comparison", type="primary", key="cmp_go")

    @st.cache_data(show_spinner=False)
    def run_cmp(_X, _y, task, models, k, key, _groups):
        return M.compare_models(_X, _y, task, models, k, groups=_groups)

    with r2:
        ck = (name2, target2, tuple(chosen), k, blocked2)
        if go_btn or st.session_state.get("cmp_key") == ck:
            st.session_state["cmp_key"] = ck
            with st.spinner("Cross-validating..."):
                cv = run_cmp(X2, y2, task2, tuple(chosen), k, (name2, target2, blocked2),
                             g2 if blocked2 else None)
            st.caption("Year-blocked CV - one fold per held-out year group." if blocked2 else "Random k-fold CV.")
            st.dataframe(M.summarise(cv), width="stretch")
            metric_cols = [c for c in cv.columns if c not in ("Model", "Fold") and "time" not in c]
            m_sel = st.selectbox("Metric to plot", metric_cols)
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(px.box(cv, x="Model", y=m_sel, color="Model", points="all",
                                       color_discrete_map=COLORS, title=f"{m_sel} across folds")
                                .update_layout(showlegend=False), width="stretch")
            with g2:
                t = cv.groupby("Model")[["Train time (s)", "Predict time (s)"]].mean().reset_index()
                st.plotly_chart(px.bar(t.melt("Model"), x="Model", y="value", color="variable", barmode="group",
                                       labels={"value": "seconds", "variable": ""}, title="Mean time per fold"),
                                width="stretch")
            st.download_button("Download fold-level results (CSV)", cv.to_csv(index=False), "rf_vs_boosting.csv")
        else:
            st.info("Pick a dataset and press **Run comparison**.")

# --------------------------------------------------------------------------- #
# 3. RF + ARIMA
# --------------------------------------------------------------------------- #
def render_ts():
    st.header("Hybrid RF + ARIMA - Sri Lankan time series")
    st.write("**Idea (Zhang, 2003):** a series = linear part + non-linear part. ARIMA captures the linear "
             "autocorrelation; a Random Forest then learns the non-linear pattern left in ARIMA's residuals. "
             "Hybrid B instead gives the RF the ARIMA forecast as an extra feature. "
             "All forecasts are one-step-ahead on a held-out test period.")
    l3, r3 = st.columns([1, 3])
    with l3:
        src = st.radio("Series", ["Weekly dengue cases (2006-2026)", "Weekly rainfall (2020-2025)",
                                  "Upload CSV (date, value)"])
        default_test = 104
        if src.startswith("Weekly dengue"):
            district = st.selectbox("District", ["All island"] + D.dengue_districts(),
                                    index=1 + D.dengue_districts().index("Colombo"))
            start_year = st.slider("Start year", 2007, 2022, 2010)
            series = D.load_dengue_series(district, f"{start_year}-01-01")
        elif src.startswith("Weekly rainfall"):
            wd = D.weather_districts()
            district = st.selectbox("District", wd, index=wd.index("Colombo"), key="rain_d")
            series = D.load_rain_series(district)
            default_test = 52
        else:
            up = st.file_uploader("CSV with a date column and a value column", type="csv", key="ts_up")
            if up is None:
                st.info("Upload a CSV to continue.")
                return
            u = pd.read_csv(up)
            dc = st.selectbox("Date column", u.columns)
            vc = st.selectbox("Value column", [c for c in u.columns if c != dc])
            series = u.set_index(pd.to_datetime(u[dc]))[vc].astype(float).sort_index()
        test_w = st.slider("Test period (last N points)", 26, 208, default_test, 13)
        n_lags = st.slider("Lag features for RF", 2, 16, 8)
        auto = st.checkbox("Auto-select ARIMA order (AIC)", value=True)
        order = None
        if not auto:
            p_, d_, q_ = st.columns(3)
            order = (p_.number_input("p", 0, 6, 2), d_.number_input("d", 0, 2, 1), q_.number_input("q", 0, 6, 1))
        log_t = st.checkbox("log(1+y) transform", value=True, help="Stabilises variance of count data.")
        horizon = st.slider("Future forecast (weeks)", 4, 26, 12)

    @st.cache_data(show_spinner="Fitting ARIMA and Random Forests...")
    def run_ts(_s, key, test_w, order, n_lags, log_t):
        r = HT.run_hybrid(_s, test_size=test_w, order=order, n_lags=n_lags, log_transform=log_t)
        fut = HT.forecast_future(r, 26)
        r.pop("_state")
        return r, fut

    with r3:
        key = (series.name, str(series.index[0]), len(series))
        res, fut = run_ts(series, key, test_w, tuple(order) if order else None, n_lags, log_t)
        best = res["metrics"]["RMSE"].idxmin()
        c1, c2, c3 = st.columns(3)
        c1.metric("ARIMA order (p,d,q)", str(res["order"]))
        c2.metric("Best model (RMSE)", best.split(":")[0])
        gain = 100 * (1 - res["metrics"].loc[best, "RMSE"] / res["metrics"].loc["ARIMA", "RMSE"])
        c3.metric("RMSE improvement over ARIMA", f"{gain:.1f}%")

        fig = go.Figure()
        hist_n = min(len(res["dates"]), res["split"] + len(res["test_dates"]))
        fig.add_scatter(x=res["dates"][:hist_n], y=res["actual_all"][:hist_n], name="Actual",
                        line=dict(color="#9E9E9E", width=1))
        show = st.multiselect("Models to plot", list(res["preds"]), default=["ARIMA", best])
        for k_ in show:
            fig.add_scatter(x=res["test_dates"], y=res["preds"][k_], name=k_, line=dict(color=COLORS.get(k_)))
        f = fut.head(horizon)
        fig.add_scatter(x=f["week_start"], y=f["Hybrid"], name="Future: Hybrid A", line=dict(dash="dot", color="#AD1457"))
        fig.add_scatter(x=f["week_start"], y=f["ARIMA"], name="Future: ARIMA", line=dict(dash="dot", color="#1565C0"))
        fig.add_vrect(x0=res["test_dates"][0], x1=res["test_dates"][-1], fillcolor="#E3F2FD", opacity=0.4,
                      line_width=0, annotation_text="test period")
        fig.update_layout(height=460, title=series.name, yaxis_title="cases / week", legend=dict(orientation="h"))
        # zoom on the test period (+ some context); the range slider lets you pan back through history
        ctx = max(0, res["split"] - len(res["test_dates"]))
        fig.update_xaxes(range=[res["dates"][ctx], f["week_start"].iloc[-1] + pd.Timedelta(days=14)],
                         rangeslider=dict(visible=True, thickness=0.06))
        st.plotly_chart(fig, width="stretch")

        m1, m2 = st.columns([3, 2])
        with m1:
            st.subheader("Test-period accuracy")
            st.dataframe(res["metrics"].style.format("{:.2f}").highlight_min(color="#C8E6C9"),
                         width="stretch")
        with m2:
            st.subheader("What the RF learned from residuals")
            st.plotly_chart(px.bar(res["resid_importance"].head(10).sort_values(), orientation="h",
                                   color_discrete_sequence=["#AD1457"]).update_layout(showlegend=False, height=300,
                                                                                       xaxis_title="importance", yaxis_title=""),
                            width="stretch")
        with st.expander("ARIMA model summary"):
            st.code(res["arima_summary"])

# --------------------------------------------------------------------------- #
# 4. RF + ANN
# --------------------------------------------------------------------------- #
def render_ann():
    st.header("Hybrid RF + ANN for tabular regression")
    st.write("**Task:** predict next week's dengue cases in each district from this week's cases and the "
             "rainfall, humidity and temperature of the previous 0-8 weeks (rain creates mosquito breeding "
             "sites, so its effect shows up weeks later). Target is log(1 + cases), so an RMSE of 0.6 means "
             "predictions are typically within a factor of about e^0.6 ≈ 1.8 of the true count.")
    st.markdown(
        "| Hybrid | How it combines RF and ANN |\n|---|---|\n"
        "| 1. Weighted average | blend both predictions, weight chosen on a validation split |\n"
        "| 2. Stacking | RF and ANN are base learners; a Ridge meta-model combines their out-of-fold predictions |\n"
        "| 3. Residual | RF predicts; the ANN learns the RF's out-of-bag errors |\n"
        "| 4. Feature selection | RF importance picks the inputs the ANN is trained on |")
    l4, r4 = st.columns([1, 3])
    with l4:
        picked = dataset_picker("ann", allow=("Dengue cases next week (regression)",))
        if picked is None:
            return
        df4, target4, name4 = picked
        X4, y4, task4 = prep(df4, target4)
        if task4 != "regression":
            st.error("This section needs a numeric (regression) target.")
            return
        st.caption(f"{X4.shape[0]:,} rows · {X4.shape[1]} features")
        h1 = st.select_slider("ANN layer 1 units", [16, 32, 64, 128], 64)
        h2 = st.select_slider("ANN layer 2 units", [0, 8, 16, 32, 64], 32)
        k4 = st.slider("CV folds", 3, 10, 3, key="k4")
        g4 = D.year_groups(df4.dropna(subset=[target4]), target4)
        blocked4 = st.checkbox("Year-blocked CV", value=g4 is not None, disabled=g4 is None, key="blk4",
                               help="Each fold holds out whole years (no leakage between districts on the same week).")
        go4 = st.button("Run RF + ANN experiment", type="primary")

    @st.cache_data(show_spinner=False)
    def run_ann(_X, _y, k, hidden, key, _groups):
        return HA.evaluate(_X, _y, k, hidden, groups=_groups)

    with r4:
        hidden = (h1,) if h2 == 0 else (h1, h2)
        k_ann = (name4, target4, k4, hidden, blocked4)
        if go4 or st.session_state.get("ann_key") == k_ann:
            st.session_state["ann_key"] = k_ann
            with st.spinner("Training 6 models x folds..."):
                folds, summ, oof = run_ann(X4, y4, k4, hidden, (name4, target4, blocked4),
                                           g4 if blocked4 else None)
            flat = pd.DataFrame({f"{m} ({s})": summ[(m, s)] for m in ["RMSE", "MAE", "R2"] for s in ["mean", "std"]})
            st.dataframe(flat.style.format("{:.3f}").highlight_min(subset=["RMSE (mean)", "MAE (mean)"], color="#C8E6C9")
                         .highlight_max(subset=["R2 (mean)"], color="#C8E6C9"), width="stretch")
            g1, g2 = st.columns(2)
            with g1:
                short = folds.assign(Model=folds["Model"].str.split(":").str[0]
                                     .replace({"Random Forest": "RF", "ANN (MLP)": "ANN"}))
                st.plotly_chart(px.box(short, x="Model", y="RMSE", points="all", title="RMSE across folds",
                                       color_discrete_sequence=["#2E7D32"])
                                .update_layout(height=430, xaxis_title=""), width="stretch")
                st.caption("Hybrids 1-4 are described in the table above.")
            with g2:
                pick = st.selectbox("Out-of-fold predictions for", list(oof.columns), index=3)
                fig = px.scatter(x=y4, y=oof[pick], opacity=0.6, labels={"x": "Actual", "y": "Predicted"},
                                 title=pick)
                lo, hi = float(y4.min()), float(y4.max())
                fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
                st.plotly_chart(fig.update_layout(height=430), width="stretch")
            st.download_button("Download fold-level results (CSV)", folds.to_csv(index=False), "rf_ann_results.csv")
        else:
            st.info("Press **Run RF + ANN experiment**.")

# --------------------------------------------------------------------------- #
# 5. Research notes
# --------------------------------------------------------------------------- #
def render_notes():
    st.header("Comparing RF with other ensemble methods")
    st.markdown("""
| | **Random Forest** (bagging) | **XGBoost / LightGBM / CatBoost** (boosting) |
|---|---|---|
| How trees are built | Independently, in parallel, on bootstrap samples | Sequentially - each tree fixes the previous trees' errors |
| Main effect | Reduces **variance** | Reduces **bias** (and variance with regularisation) |
| Tuning needed | Little - good out of the box | More (learning rate, depth, n_estimators, regularisation) |
| Overfitting with more trees | No (error plateaus) | Yes, without early stopping / small learning rate |
| Free validation | Out-of-bag (OOB) error | Needs a validation set |
| Categorical features | Need encoding | CatBoost handles natively; LightGBM supports them |
| Speed | Parallel, but deep trees are memory-heavy | LightGBM fastest (histogram, leaf-wise); XGBoost close |
| Typical accuracy on tabular data | Strong baseline | Usually slightly higher when tuned |

**Tabular vs image/text data**
* **Tabular** - tree ensembles are the state of the art: they handle mixed feature types, different scales,
  missing values and non-linear interactions without feature scaling.
* **Image / text** - raw pixels or tokens have spatial/sequential structure that trees cannot exploit
  (no notion of locality or word order); CNNs / Transformers dominate. Trees become useful again only on top of
  extracted features or embeddings.

**Why these Sri Lankan datasets?**
* **Rain tomorrow** - agriculture and disaster preparedness depend on short-range rain forecasts, and the
  two monsoons make rainfall strongly seasonal and district-specific.
* **Dengue** - one of Sri Lanka's largest public-health burdens. Cases follow rainfall and humidity with a
  delay of weeks (rain → breeding sites → mosquitoes → infections), which gives a natural use for lagged
  weather features and for time-series models.

**Why hybrids?**
* **RF + ARIMA** - ARIMA is linear and models autocorrelation well; RF captures non-linear effects
  (e.g. seasonal outbreaks) but cannot extrapolate trends. Combining them covers both weaknesses.
* **RF + ANN** - RF is robust on small, noisy tabular data; ANNs learn smooth functions and can extrapolate.
  Their errors are only partly correlated, so combining them (stacking) can reduce error.
""")


for _tab, _render in zip((tab1, tab2, tab3, tab4, tab5),
                         (render_rf, render_cmp, render_ts, render_ann, render_notes)):
    with _tab:
        _render()
