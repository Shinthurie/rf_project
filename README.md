# Random Forest Dashboard: Hands-on Project (Sri Lankan data)

Index: 235521T

Every part of the project uses **local Sri Lankan data**.

| Slide item | Local dataset used | Where |
|---|---|---|
| Build an RF dashboard | All three datasets below | `app.py` → **RF Explorer** tab |
| Compare RF vs XGBoost on local datasets | ① **Rain tomorrow?** (25 districts, daily) ② **High-risk dengue week ahead?** (25 districts, weekly) | **RF vs Boosting** tab, Exp. 1 (LightGBM and CatBoost included too) |
| RF + ARIMA for time series | **Weekly dengue cases**, 26 districts, 2006 to 2026 (and weekly rainfall in the dashboard) | **RF + ARIMA** tab, Exp. 2 |
| RF + ANN for tabular regression | **Next week's dengue cases** from lagged rainfall, humidity and temperature | **RF + ANN** tab, Exp. 3 |
| Research: RF vs boosting; tabular vs image/text | n/a | **Research notes** tab |

## Run it

```bash
pip install -r requirements.txt
python build_datasets.py      # optional - rebuilds data/*.csv from data/raw/ (already included)
streamlit run app.py          # dashboard at http://localhost:8501
python run_experiments.py     # regenerates every table/figure in results/ (~10 min)
```

## Data

**Raw sources (`data/raw/`)**

| File | Content | Source |
|---|---|---|
| `srilanka_dengue_weekly.csv` | Weekly dengue cases for every district, Dec 2006 to Sep 2026 | Epidemiology Unit / National Dengue Control Unit, Ministry of Health, compiled in the `denguedatahub` R package (Talagala et al.) |
| `srilanka_weather_daily_2020_2025.csv` | Daily mean temperature, humidity, rainfall and wind speed for 25 districts, Jan 2020 to Sep 2025 | Public GitHub dataset *SriLanka-Weather-Dataset* (Lahirulakshan129). **Check with your lecturer how to cite it, since the original weather provider is not stated.** |

**Modelling tables built by `build_datasets.py`**

| Table | Rows | Target | Features |
|---|---|---|---|
| `sl_rain_tomorrow.csv` | 50,967 district-days | rain > 1 mm tomorrow (44% yes) | today's and the last 2 days' weather, 3/7-day rain totals, season, district |
| `sl_dengue_weather_weekly.csv` | 7,051 district-weeks | `high_risk_next_week`: cases above the district's 75th percentile (23% yes); `log_cases_next_week` | cases from the last 4 weeks; rainfall lagged 0 to 8 weeks; humidity and temperature lagged 0 to 6 weeks; season; district |

Cleaning: 364 weather rows with no date were removed and duplicate district-days averaged. District names were matched between sources (Hambanthota → Hambantota, NuwaraEliya → Nuwara Eliya). Kalmunai has no weather data, so it is excluded from the joined table.

## Evaluation design: why "year-blocked" CV

The tabular data is a **panel**: 25 districts observed on the same days. With ordinary random k-fold CV, Colombo on a given day can be in the training set while Gampaha on the same day is in the test set. Neighbouring districts share almost the same weather, so the model can effectively look up the answer.

**Year-blocked CV** holds out whole calendar years in each fold, which is an honest "predict an unseen period" test. All main results use it, and Exp. 1 also reports random CV to show the difference. The RF Explorer tab trains on 2020–2024 and tests on 2025 by default.

Every model gets the same folds and the same budget of 300 trees, with untuned defaults.

## Results

### Exp. 1: RF vs gradient boosting (year-blocked CV, mean of folds)

**① Rain tomorrow (50,967 district-days)**

| Model | Accuracy | F1 | ROC-AUC | Train time / fold |
|---|---|---|---|---|
| CatBoost | **0.732** | **0.695** | **0.803** | 2.2 s |
| Random Forest | 0.731 | 0.689 | 0.797 | 12.4 s |
| LightGBM | 0.729 | 0.690 | 0.799 | 0.9 s |
| XGBoost | 0.727 | 0.689 | 0.797 | 1.2 s |

**② High-risk dengue week ahead (7,051 district-weeks)**

| Model | Accuracy | F1 | ROC-AUC | Train time / fold |
|---|---|---|---|---|
| LightGBM | **0.839** | **0.578** | 0.844 | 0.4 s |
| XGBoost | **0.839** | **0.578** | **0.845** | 0.6 s |
| CatBoost | 0.833 | 0.550 | 0.844 | 1.0 s |
| Random Forest | 0.811 | 0.455 | 0.806 | 2.2 s |

**Random vs year-blocked CV (rain tomorrow, accuracy):**

| Model | Random CV | Year-blocked CV | Drop |
|---|---|---|---|
| Random Forest | 0.769 | 0.731 | **−3.8 pts** |
| XGBoost | 0.753 | 0.727 | −2.6 |
| LightGBM | 0.749 | 0.729 | −2.0 |
| CatBoost | 0.743 | 0.732 | −1.1 |

Findings:
- On rain prediction the four models are effectively **tied** (within 0.5 pts). On dengue risk, **boosting beats RF** clearly, especially on F1 (0.58 vs 0.46) for the minority "high-risk" class.
- Under random CV, RF appeared to be the best model for rain. That advantage was leakage: deep RF trees memorise near-duplicate rows from neighbouring districts on the same day, so they gain the most from it. This is a useful methodological point for the report.
- RF is the **slowest** to train here; LightGBM is the fastest.

### Exp. 2: RF + ARIMA (weekly dengue cases, last 104 weeks held out, one-step-ahead)

RMSE improvement over ARIMA alone:

| District | Hybrid A: ARIMA + RF(residuals) | Hybrid B: RF with ARIMA feature | RF alone |
|---|---|---|---|
| All island | **+25.5 %** | +15.1 % | +7.7 % |
| Colombo | +1.2 % | **+21.0 %** | +13.9 % |
| Gampaha | +7.1 % | **+9.1 %** | +2.5 % |
| Kandy | **+18.3 %** | +14.9 % | +12.5 % |
| Kalutara | **+8.1 %** | +0.9 % | −7.7 % |
| Kurunegala | **+6.4 %** | −6.5 % | −14.3 % |
| Jaffna | **+2.5 %** | −1.1 % | −0.9 % |

Hybrid A beats ARIMA in **all 7 series** and is the best model in 5 of 7. RF alone is unreliable: it is worse than ARIMA in 3 of 7 districts. The **combination** is what helps: ARIMA handles linear autocorrelation, and RF adds the non-linear correction.

### Exp. 3: RF + ANN (next week's dengue cases, year-blocked CV)

| Model | RMSE (log) | R² | MAE (cases) |
|---|---|---|---|
| Hybrid 2: Stacking | 0.611 | 0.794 | **9.6** |
| Hybrid 1: Weighted average | **0.610** | **0.795** | 9.9 |
| Random Forest | 0.614 | 0.793 | 10.0 |
| Hybrid 3: RF + ANN on residuals | 0.621 | 0.788 | 10.3 |
| Hybrid 4: RF features → ANN | 0.651 | 0.768 | 11.1 |
| ANN alone | 0.651 | 0.768 | 11.1 |

- Stacking and weighted averaging beat both models alone; stacking cuts the error by about 4% in case counts. The gain is small but consistent.
- The ANN alone is clearly weaker than RF on this tabular data, the usual pattern.
- **Feature importance** (`results/exp3_feature_importance.csv`): recent case counts dominate, at about 86% of total importance for the last 4 weeks. Each weather feature contributes under 1%. Among them, the top-ranked are the **8-week rainfall total and rainfall/humidity 2 to 4 weeks earlier**, and same-week rainfall ranks lowest of the rainfall lags. That fits the mosquito breeding delay, but the effect is small; temperature matters least.

## Limitations / extensions
- **Weather source:** the daily weather file comes from a public GitHub repository whose original provider is not documented. Ideally, verify it against Department of Meteorology data.
- **Dengue reporting gaps:** some weeks show 0 cases for large districts (e.g. Colombo, April and June 2026). These are probably reporting gaps; treating them as missing would help all models.
- **No tuning:** with tuning (e.g. Optuna), boosting usually gains more than RF does.
- **Uneven folds:** year-blocked folds differ a lot (e.g. 2020 was a COVID year with unusual reporting), which explains the large standard deviations in Exp. 1 ②.
- **Possible extension:** add weather as exogenous inputs to the time-series model (ARIMAX + RF).
