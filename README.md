# Probabilistic Day-Ahead Price Forecasting for Türkiye (EPİAŞ)

> Status: learning/portfolio project; not deployed.

## 1. What this project does

Türkiye's day-ahead electricity market publishes one price per hour, called PTF (market clearing price, TL/MWh). This project forecasts tomorrow's 24 hourly prices, as of 10:00 the day before, and gives:

- a **point forecast** (one number per hour), and
- a **prediction interval** per hour (80% and 90% target coverage).

It is a learning and portfolio project. It is **not** a live trading tool and it is **not deployed**.

## 2. Results

MAE = mean absolute error in TL/MWh (lower is better). rMAE = MAE divided by the MAE of the "yesterday, same hour" baseline (D-1); below 1 means better than D-1.

### 2.1 Point forecast, test period T (2025-10-21 to 2026-10-20, 8,760 hours)

Walk-forward evaluation: the model is retrained every 28 days on an expanding window [N8].

| Model | MAE | rMAE |
|---|---|---|
| Yesterday, same hour (D-1) | 519.2 [N1] | 1.000 |
| Same weekday last week (D-7) | 598.6 [N1] | 1.153 |
| Simple rule (D-1 or D-7 by weekday) | 497.5 [N1] | 0.958 |
| LightGBM, no weather (first model) | 433.5 [N2] | 0.835 |
| LightGBM, with weather forecasts (first model) | 419.3 [N2] | 0.807 |
| Final model (weather + holiday features) | 420.5 [N3] | 0.810 |

The final model is **not better** than the first weather model on T (420.5 vs 419.3). The final feature set was chosen on a separate validation year V (2024-10-21 to 2025-10-20), where it scored 271.20 against 393.45 for D-1 [N4]. Hyperparameter tuning with Optuna was tried on V and **not adopted**: the gain was 1.34%, below the 1.5% rule written beforehand [N5].

A short check on 9 newer days (after T): final model 433.9, D-1 495.2 [N6]. Nine days prove nothing.

### 2.2 Prediction intervals (V and T side by side)

Two methods: **B0** = calibration on the last 56 days, fixed per 28-day block; **B3** = daily-updated calibration plus ACI (adaptive conformal inference, gamma 0.05), which adjusts the error level day by day. Width is the mean width after clipping to [0, daily price cap], in TL/MWh. The method was chosen on V before T was run; T was run once [N7].

| Method | Level | V coverage | V width | T coverage | T width |
|---|---|---|---|---|---|
| B0 | 80% | 75.43% | 829.3 | 71.50% | 1297.9 |
| B0 | 90% | 84.53% | 1108.8 | 84.45% | 1719.8 |
| B3 | 80% | 79.86% | 886.9 | 79.99% | 1548.8 |
| B3 | 90% | 89.92% | 1190.9 | 90.02% | 1942.4 |
[N7, recomputed from the stored prediction tables]

On T, B3 hit both targets (within 0.01 and 0.02 percentage points; the pre-set information threshold was 2.0 points). The B0 interval undercovers on T, as on V. **The B3 intervals on T are much wider than on V** (about 1.75x at 80% and 1.63x at 90%, computed from the table). The reason is unknown (see Limitations). Coverage is weaker than the headline in some months, for example 2026-02 at the 90% level (86.76%) [N9].

## 3. Architecture

```
EPİAŞ hourly data (price, consumption, load plan)   Open-Meteo weather FORECASTS (8 cities)
                 \                                   /
                  -> hourly table -> 18 features [N10]
                                       |
                    LightGBM point model (L1 loss, 400 trees) [N11]
                    LightGBM quantile models -> conformal intervals (CQR, daily calibration, ACI)
                                       |
                    Frozen model package  ->  FastAPI service (replay mode only)
```

- **Prices and lags.** Only information known the day before: price at the same hour 1, 2 and 7 days earlier and the previous day's mean price. Consumption and load plan are collected but not used as features, because their publication time is unknown.
- **Weather.** The features are weather *forecasts* (Open-Meteo "previous runs", `_previous_day2`), not observed weather. Whether those forecasts are really available at 10:00 the day before is **unknown**.
- **Calendar and holidays.** Hour, weekday, month, weekend flag; holiday, holiday-eve, bridge-day flags; distance to nearest holiday.
- **Missing values are never filled**; LightGBM handles them.
- **Price cap.** The official maximum price changes: 3,000 TL/MWh from 2024-07-01, 3,400 from 2025-04-05, 4,500 from 2026-04-04 [N12]. Whether a later decision exists is unknown.

## 4. How to run

You need Python 3.13 and the packages in `requirements.txt`. The data and the trained model package are **not** in this repository.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

The unit tests: 189 ran and passed on the author's machine, including tests that need the real data [N13]. Tests that need the real data or the model package are skipped on a machine without them (in the latest CI run, 189 tests ran and 10 were skipped [N13]).

Start the service (replay mode) once you have a model package and a table:

```bash
export MODEL_DIR=<folder of the model package>
export DATA_PATH=<path of the hourly table, parquet>
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/predict?date=2026-10-01"
```

`/predict` returns 24 hourly entries (point, 80% and 90% interval, the level used that day, and the actual price for information only). Dates before 2026-09-22 return HTTP 400 (the model saw that day in training), dates after the last table day return 404, a bad date format returns 422. The service answers for 2026-09-22 to 2026-10-29 only (38 days, 912 hours) [N14]. **There is no live data pipeline: it cannot forecast tomorrow.**

To rebuild the hourly table you need your own account on the EPİAŞ transparency platform; credentials go in a local `.env` (see `.env.example`; the repository contains none). The full download pipeline was not re-run for this README.

### 4.1 Dashboard (Streamlit)

`app/streamlit_app.py` is a small dashboard that shows **stored results of the test period T only** (2025-10-21 to 2026-10-20). For a chosen day it shows the hourly actual price, the final point forecast, the B3 prediction interval (80% and 90%) and the D-1, D-7 and rule baselines. It also shows a summary for the chosen day and for the whole of T (MAE, B3 coverage, mean interval width). It is **not live**, it does **not produce forecasts**, and it does **not connect to EPİAŞ**: it only reads three stored files.

Those three files are **not in this repository** (they are git-ignored): `data/processed/final_tahminleri.parquet`, `data/processed/oturum_T_final/aralik_T_tahminleri.parquet` and `reports/baselines_gunluk.csv`. If any of them is missing, the dashboard shows a clear error message listing the missing paths and stops.

Run it in its own environment; the packages are listed in `requirements-app.txt` and the main environment is not touched:

```bash
python -m venv .venv-app
# activate .venv-app, then:
pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py
```

T was seen before, so what the dashboard shows is **not a clean final exam**. The dashboard has **not been deployed**: it is not on the internet.

## 5. Docker and CI status

- **CI (GitHub Actions)** has two jobs, `test` and `docker` [N13].
  - `test` runs the unit tests on Python 3.13: 189 tests ran, 10 were skipped (they need the real data or the model package) [N13].
  - `docker` builds `docker/Dockerfile` on GitHub's Linux machine (run 42368669858). The image is 690 MB. Inside the image, `lightgbm` 4.7.0 and `pandas` 3.0.6 import correctly. With `MODEL_DIR` and `DATA_PATH` unset, the container refuses to start (exit code 3) with the message "MODEL_DIR ve DATA_PATH ortam degiskenleri gerekli" [N13].
  - An informational experiment showed that `lightgbm` could **not** be imported without the `libgomp1` system package (`libgomp.so.1: cannot open shared object file`), so `libgomp1` is needed (tested on `python:3.13-slim` with `lightgbm` 4.7.0) [N15].
- **What was NOT tested:** `/predict` was **not** tried inside the image with a model and data (CI has neither). The image has **never been run on the development machine** (Docker is not installed there). The image was **not uploaded** anywhere (no registry). There is **no cloud deployment**.
- **Streamlit dashboard**: a local dashboard of stored T-period results exists (section 4.1); it has not been deployed.

## 6. Limitations (please read)

- **The test period was seen before.** The first LightGBM model was evaluated on T. The final result, including the T interval run, is therefore **not a clean final exam**. The final configuration was chosen on V only.
- **One period, one run, no significance test.** T's interval run was done once. No statistical test of the V-vs-T differences was done.
- **Interval width on T is much larger than on V**; the reason is unknown. Price-cap changes are a hypothesis that was **not tested**.
- **Weak spots.** Hours with price 0 (409 hours): the final model is worse than D-1 (MAE 244.0 vs 191.6). April 2026 (587.6 vs 536.4) and May 2026 (399.8 vs 341.4) are also worse than D-1 [N3]. On V, the 56-day interval covered only 44.5% in block 7 (2025-04-07 to 2025-05-04); the cause is **unknown**. The price-cap hypothesis was not tested [N7].
- **Input timing is unverified.** Whether `_previous_day2` weather forecasts are ready at 10:00 the day before is unknown; the publication time of the load plan is unknown; whether the bridge-day flag is known at forecast time was not verified.
- **Replay only**, no live pipeline. The service answers only for the 38 replay days.
- **Raw EPİAŞ data and trained models are not in the repository.** Holiday bridge days come from news sites (medium confidence). The last price-cap decision found is dated 2 April 2026.
- **Quantile outputs can cross.** The code uses the raw (unsorted) quantiles and applies no sorting fix. On T: q10 above q90 in 36 of 8,760 hours, q05 above q95 in 10, any ordering violation in 2,027 (23.1%) [N7]. In the stored final intervals no lower bound is above its upper bound, but the 90% interval does not fully contain the 80% interval in 1,399 of 8,760 hours (16.0%) for B3 on T (1,055 hours, 12.0%, on V).
- **Licence:** the code is released under the MIT licence (see the `LICENSE` file and the "License" section below).

## 7. Data and licence notes

- **EPİAŞ data** is not redistributed here. No clear redistribution terms were found, so the repository contains only download code and two small reference tables (price caps, holidays).
- **Weather data**: Weather data by Open-Meteo.com, licensed CC BY 4.0 (non-commercial use of the free API); attribution required if you reuse it.
- **Price-cap table**: compiled from EPİAŞ board decision announcements; sources listed in `reports/tavan_arastirma.md`.

## 8. Repository layout

```
src/data/       download and table building (prices, consumption, weather, holidays, price caps)
src/features/   feature building
src/models/     baselines, LightGBM, tuning, conformal intervals, evaluation scripts
src/serving/    model package builder, predictor, FastAPI service
tests/          unit tests (synthetic data) and tests that need the real data
reports/        written reports for each experiment
.github/        CI workflow (GitHub Actions)
docker/         Dockerfile (built in CI, not run with a model)
app/            Streamlit dashboard (stored T-period results only)
requirements-app.txt   packages of the dashboard (separate environment, `.venv-app`)
requirements-lock.txt  `pip freeze` of the main environment (43 lines); the Docker image does not use it (it installs `requirements.txt`)
```

## 9. Next steps

1. A cloud deployment (the Streamlit dashboard exists but has not been deployed).
2. A live data pipeline (not planned in detail).

## 10. Sources and credits

- Market data: EPİAŞ transparency platform (via the `eptr2` Python package).
- Weather forecasts: Open-Meteo.com (CC BY 4.0).
- Methods: LightGBM; conformalized quantile regression (CQR); adaptive conformal inference (ACI).

## License

The code in this repository is released under the MIT licence; see the `LICENSE` file. The Open-Meteo CC BY 4.0 attribution in section 7 still applies to any weather data you obtain from Open-Meteo. EPİAŞ data is **not** covered by this licence: it is not in the repository and is not redistributed.

---

## Source table

All sources are in the project folder; full detail in `data/processed/oturum_hafta8/sayilar.md`.

| id | numbers | file | how checked |
|---|---|---|---|
| N1 | 519.2, 598.6, 497.5 | reports/baselines.md; reports/final_model_raporu.md "Toplam" | read from file |
| N2 | 433.5, 419.3 | reports/final_model_raporu.md | recomputed from lgbm_tahminleri.parquet |
| N3 | 420.5; 244.0/191.6; 587.6/536.4; 399.8/341.4; 409 zero-price hours | reports/final_model_raporu.md | 420.5 recomputed from final_tahminleri.parquet; others read from file |
| N4 | 271.20, 393.45, V 2024-10-21..2025-10-20 | reports/dogrulama_deneyleri.md | read from file |
| N5 | Optuna 271.200 -> 267.561 (+1.34%), threshold 1.5% | reports/optuna_dogrulama.md | read from file |
| N6 | 433.9, 495.2, 9 days | reports/yeni_gun_degerlendirme.md | read from file |
| N7 | all interval numbers in 2.2; block 7 44.49%; crossing 36 | reports/aralik_final_t.md; reports/aralik_gunluk.md; reports/probabilistic.md | coverage and width recomputed from the stored prediction parquet files, identical |
| N8 | 8,760 hours per period; 28-day blocks; training start 2024-03-08 | reports/final_config.json; reports/lgbm_ilk_model.md | read from file; row counts from parquet |
| N9 | 2026-02: B3 90% 86.76% | reports/aralik_final_t.md "Ay başına" | read from file |
| N10 | 18 features | reports/final_config.json | read from file |
| N11 | L1 loss, 400 trees | reports/final_config.json | read from file |
| N12 | 3,000 / 3,400 / 4,500 and dates | reports/tavan_arastirma.md | read from file |
| N13 | CI: 2 jobs (test, docker); 189 tests, 10 skipped; docker job built the image (run 42368669858), 690 MB, lightgbm 4.7.0, pandas 3.0.6, exit code 3 without env vars; 189 tests also ran locally | data/processed/oturum_docker/ci_push_rapor.md ("CI sonucu"); .github/workflows/ci.yml lines 42-88 | read from file. The CI numbers come from Claude Code's `gh` output and can also be seen on the GitHub Actions page |
| N14 | 2026-09-22..2026-10-29, 38 days, 912 hours; HTTP 400/404/422 | model package info file; src/serving/api.py | read from file |
| N15 | libgomp1 needed: lightgbm import failed without it; image never run locally or with a model | data/processed/oturum_docker/ci_push_rapor.md ("libgomp1 deneyi"); data/processed/oturum_docker/faz1_rapor.md | read from file. From Claude Code's `gh` output; also visible on the GitHub Actions page |
