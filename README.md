# Weather Intelligence – Experimentation Repository

**Student Name:** Nishaanth Govindaraj
**Student ID:** 26044382
**Subject:** 36120 Advanced Machine Learning Application – Assessment 2

Experimentation repository for predicting two custom weather targets for Sydney (lat -33.8688, lon 151.2093)
from the Open-Meteo Historical Weather API:

| Target | Type | Horizon | Deployed model |
|---|---|---|---|
| Climate Comfort Index (CCI, 0-100) | Regression | Next 3 days | Ridge regression (one model per day) |
| Weather Hazard Category (WHC, 4 classes) | Multiclass classification | Exactly +7 days | XGBoost with a hazard alert threshold |

Related repositories:
- Custom package (`my_krml_26044382`, v0.0.5 on TestPyPI): https://github.com/nishaanth-20/36120-26SP-group6-26044382-package
- FastAPI deployment repository: https://github.com/nishaanth-20/adv_mla_at2_api
- Live API (Render): https://adv-mla-at2-api-vxja.onrender.com (interactive docs at `/docs`)

---

## Project structure

Based on the Cookiecutter Data Science template.

```
├── notebooks/
│   ├── comfort_climate/     CCI experiments 1-3 (Ridge, XGBoost, Optuna XGBoost + final test)
│   └── weather_hazard/      WHC experiments 1-3 (Logistic regression, XGBoost, threshold models + final test)
├── models/
│   ├── comfort_climate/     trained CCI models and cci_metadata.json
│   └── weather_hazard/      trained WHC models and whc_metadata.json
├── reports/
│   ├── html/                executed notebooks exported to HTML
│   └── *.csv                validation and test results of each experiment
├── data/                    not tracked by Git, created by the notebooks (see below)
├── pyproject.toml           dependencies (managed with uv)
└── README.md
```

---

## Setup

### 1. Install uv
Follow https://docs.astral.sh/uv/getting-started/installation/

### 2. Clone the repository and install the dependencies
```bash
git clone https://github.com/nishaanth-20/adv_mla_at2_experiments.git
cd adv_mla_at2_experiments
uv sync
```
`uv sync` installs Python 3.12.13 (pinned in `.python-version`) and all dependencies, including the custom
package `my-krml-26044382` from TestPyPI (the index is configured in `pyproject.toml`).

### 3. Launch Jupyter Lab
```bash
uv run jupyter lab
```
(Or open the notebooks in VS Code and select the `.venv` kernel of this project.)

### Windows note
Each notebook sets `OPENBLAS_NUM_THREADS=4` before importing numpy. This avoids an OpenBLAS memory allocation
error on machines with many CPU threads. If the error still appears, set it to `1`.

---

## Running the notebooks

Run the notebooks **in this order**. Later experiments read results and models saved by earlier ones.

| Order | Notebook | What it does | Approx. time |
|---|---|---|---|
| 1 | `comfort_climate/..._cci_experiment_1.ipynb` | Downloads the data (first run only), Ridge regression, baselines | 2-3 min (first run) |
| 2 | `comfort_climate/..._cci_experiment_2.ipynb` | XGBoost regressor | 3-5 min |
| 3 | `comfort_climate/..._cci_experiment_3.ipynb` | Optuna XGBoost + sample weights, test evaluation, **deployment models** | 5-10 min |
| 4 | `weather_hazard/..._whc_experiment_1.ipynb` | Logistic regression with class weights, baselines | 1 min |
| 5 | `weather_hazard/..._whc_experiment_2.ipynb` | XGBoost classifier with class weights | 3-5 min |
| 6 | `weather_hazard/..._whc_experiment_3.ipynb` | Optuna models + alert threshold, test evaluation, **deployment model** | 5-10 min |

**Data:** the first notebook downloads hourly observations for 2000-2025 from the Open-Meteo Historical Weather
API (one request per year, with automatic retries) and caches them in `data/raw/yearly/`. Later notebooks reuse
the cache. Data from 2026 onwards is never downloaded; it is reserved for production.

**Data split** (chronological, identical in all experiments): training 2000-2019, validation 2020-2022,
test 2023-2025, with a gap equal to the forecast horizon between the sets.

---

## Results summary (test period 2023-2025)

### Climate Comfort Index – RMSE (lower is better)

| Model | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Ridge regression (deployed)** | **8.39** | **9.48** | **9.84** |
| XGBoost (Optuna, weighted) | 8.85 | 10.33 | 10.80 |
| Persistence baseline | 9.80 | 12.06 | 12.89 |
| Monthly climatology baseline | 10.15 | 10.15 | 10.14 |

### Weather Hazard Category – 7 days ahead

| Model | Macro F1 | Hazard recall |
|---|---|---|
| **XGBoost with alert threshold (deployed)** | 0.259 | 0.022 |
| Logistic regression with alert threshold | 0.103 | 0.756 (alerts on 80% of days) |
| Monthly mode baseline | 0.266 | 0.000 |
| Persistence baseline | 0.243 | 0.000 |

The CCI models beat both baselines at every horizon. The 7-day hazard category could not be predicted
reliably from local weather history: the models rank hazardous days only slightly better than chance. The
deployed WHC model should be treated as a low-confidence indication of the week's general risk level. Full
analysis is in the notebooks, the HTML exports in `reports/html/`, and the final report.

---

## Deployed model artefacts

| File | Used by the API for |
|---|---|
| `models/comfort_climate/cci_ridge_cci_t_plus_1.joblib`, `_2`, `_3` | `/predict/index/comfort_climate` |
| `models/comfort_climate/cci_metadata.json` | `/model-metadata` |
| `models/weather_hazard/whc_xgb.joblib` | `/predict/category/weather_hazard` |
| `models/weather_hazard/whc_metadata.json` | `/model-metadata` (includes the alert threshold) |

Other files in `models/` are artefacts of the earlier experiments, kept for reproducibility.

---

## AI usage

Generative AI (Claude) was used to support coding, debugging and drafting explanations. All code and analysis
were written, run and validated by the student.