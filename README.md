# Flood Prediction in Malawi

This repository is Samuel Mensah's notebook study of grid-level flood prediction. The table in `data/Train.csv` covers 16,466 squares. Each row has coordinates (`X`, `Y`), elevation, a land-cover class (`LC_Type1_mode`), and 17 weeks of precipitation for two seasons: late 2014 through early 2015, and early 2019. The training label is `target_2015`, a continuous value from 0 to 1 (mean 0.077; 13,701 squares are exactly 0). The goal is to predict `target_2019` for the same squares. Two notebooks compare regressors, engineer rainfall and terrain features, and write prediction files in the submission schema (`Square_ID`, `target_2019`).

## Approach

**`solution1.ipynb` builds a season-aligned baseline and compares models.** The 2019 precipitation columns are held out as the scoring frame. The 2014–2015 precipitation columns, together with coordinates, elevation, land cover, and `target_2015`, form the training frame. Weekly columns are renamed `week_1_precip` … `week_17_precip` so the two seasons share a schema, then aligned on column name. Nine regressors are compared with 10-fold cross-validation (`KFold`, 10 splits, `shuffle=True`, `random_state=101`) using root mean squared error. XGBoost, CatBoost, and gradient boosting are then fit on the full training frame and written to `submissions/`.

**`1.ipynb` iterates on features with LightGBM.** It keeps the same season split, then concentrates on the heavier rainfall weeks (weeks 6–9):

- Two-week precipitation sums, the maximum of those sums, the week index of the peak, and binary flags for week-over-week increases.
- Outlier removal with `EllipticEnvelope` (`contamination=0.17`, `random_state=0`), which leaves 13,666 training rows.
- A coordinate join to an external soil table, `ss.csv`. That file is not in this repository.
- Broader rainfall features: total, mean, standard deviation, skew, and coefficient of variation; acceleration; a weighted momentum term; and weeks since the peak.
- Spatial features: distance from the mean coordinate, elevation percentile, normalized coordinates, and diagonal distance.
- Interactions of elevation and land cover with precipitation, plus binary risk flags (high precipitation, consistent heavy rain, low elevation combined with high precipitation, and a recent peak).

LightGBM is trained with `learning_rate=0.07`, `max_depth=8`, and 221 trees. Validation is the mean RMSE over five 80/20 splits. Predictions are shifted by −0.08 and clipped to [0, 1] before they are saved. A later pass ranks features by a LightGBM importance fit, compares subsets, and adds candidate features one at a time. Retraining the top 10 features with the same five-split, 221-tree setup produced a slightly higher mean RMSE than the earlier engineered run (0.102042 versus 0.101604). The closing commit message is "less is more".

`test.ipynb` is a one-cell scratch notebook. It is not part of the modeling pipeline.

## Results recorded in the notebooks

These figures are the scores printed by the notebooks or stored in `catboost_info/`. The repository does not contain a leaderboard submission score.

### 10-fold cross-validation (`solution1.ipynb`)

Default hyperparameters, except `XGBRegressor(objective="reg:squarederror")` and `CatBoostRegressor(logging_level="Silent")`. Lower is better.

| Model | Mean RMSE | Std |
| --- | ---: | ---: |
| Random forest | 0.090977 | 0.005504 |
| XGBoost | 0.093132 | 0.006131 |
| CatBoost | 0.096659 | 0.006016 |
| Decision tree | 0.121933 | 0.004715 |
| Gradient boosting | 0.122426 | 0.005538 |
| k-nearest neighbors | 0.126652 | 0.007075 |
| AdaBoost | 0.174741 | 0.013595 |
| SVR | 0.197289 | 0.005109 |
| Linear regression | 0.208893 | 0.006435 |

Random forest had the lowest cross-validation RMSE. The files written under `submissions/` are the XGBoost, CatBoost, and gradient-boosting fits.

CatBoost's own training log (`catboost_info/learn_error.tsv`) runs 1,000 iterations. Training-set RMSE falls from 0.220791 at iteration 0 to 0.076810 at iteration 999. That log is in-sample learn-set error, separate from the cross-validation table above.

### LightGBM holdouts (`1.ipynb`)

Five 80/20 splits, 221 trees, `learning_rate=0.07`, `max_depth=8`. The notebook prints the label `Mean RMSLE`; the function it calls is root mean squared error.

| Run | Mean RMSE | Std | Output file |
| --- | ---: | ---: | --- |
| Engineered features, after outlier filtering and the soil join | 0.101604 | 0.003328 | `submissions_1/engineered.csv` |
| Top 10 features by importance, same training setup | 0.102042 | 0.003475 | `submissions_1/optimized_features.csv` |
| Baseline rainfall, location, and slope features only (soil file left out) | 0.103611 | 0.003779 | `submissions_1/baseline_only.csv` |

A shorter check (three 80/20 splits, 100 trees) on the filtered matrix compared feature subsets:

| Feature set | Count | Mean RMSE | Std |
| --- | ---: | ---: | ---: |
| Baseline (land cover, coordinates, elevation, weeks 7–9, max two-week sum, peak index, slope flags) | 11 | 0.110091 | 0.004150 |
| Baseline plus total precipitation, momentum, and elevation × total precipitation | 14 | 0.108987 | 0.002700 |
| Top 15 by importance | 15 | 0.107552 | 0.002547 |
| Top 10 by importance | 10 | 0.107443 | 0.002058 |

The highest-importance columns in that fit were `Y`, `diagonal_distance`, `distance_from_center`, `X`, `elevation`, and the elevation and land-cover interaction terms. Adding `total_precip`, `precip_momentum`, `elevation_total_precip`, `distance_from_center`, `precip_std`, or `days_since_peak` one at a time did not beat the baseline score from the 221-tree run (0.103611). Those single-feature checks used the 100-tree setup, so their absolute RMSE is not comparable to the 221-tree table.

`submissions_1/lili.csv` is a fourth prediction file stored in the repo. Neither notebook writes it.

## Repository layout

```
data/Train.csv                         16,466 squares × 40 columns
data/SampleSubmission.csv              Square_ID, target_2019 template
solution1.ipynb                        season alignment, 9-model comparison, baseline files
1.ipynb                                feature engineering, LightGBM, feature-set checks
test.ipynb                             scratch notebook
submissions/                           xgboost, catboost, and gradientboost baselines
submissions_1/                         engineered, optimized, baseline-only, and lili files
catboost_info/                         CatBoost training log (RMSE and TensorBoard events)
```

`data/Train.csv` has no missing values and no duplicate rows. `SampleSubmission.csv` is a template, not a model output.

## Tech stack

Python notebooks (Python 3.12 in the saved runs), Jupyter, pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, SciPy, and Matplotlib. `solution1.ipynb` also uses `requests` to download the tables.

Versions recorded in the notebook outputs from the machine that produced these runs: pandas 2.2.2, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, LightGBM 4.6.0, CatBoost 1.2.8, XGBoost 3.0.2. The repository does not pin those versions.

## Getting started

Prerequisites are Python 3 with Jupyter and the libraries imported above.

The notebooks do not read environment variables. There is no dependency file, build step, or test suite.

Local copies of the modeling table and the submission template are already in `data/`:

- `data/Train.csv`
- `data/SampleSubmission.csv`

`solution1.ipynb` does not read those local files. It downloads the training table and the sample submission from the Google Drive URLs hardcoded near the top of the notebook. `1.ipynb` reads `data/Train.csv` and `data/SampleSubmission.csv`, then reads `ss.csv` from the working directory before the later baseline-only section. `ss.csv` is not committed, so a top-to-bottom run of `1.ipynb` stops at that cell unless the file is supplied locally.

From the repository root, with those packages installed:

```bash
jupyter notebook solution1.ipynb
jupyter notebook 1.ipynb
```

Running the CatBoost section of `solution1.ipynb` rewrites `catboost_info/`. The model loop in `solution1.ipynb` overwrites `submissions/xgboost_baseline.csv`, `submissions/catboost_baseline.csv`, and `submissions/gradientboost_baseline.csv`. The LightGBM cells in `1.ipynb` overwrite `submissions_1/engineered.csv`, `submissions_1/optimized_features.csv`, and `submissions_1/baseline_only.csv`.

## Project status

Experimental notebook project. The history is three commits on 1–2 July 2025: a starter comparison (`solution1.ipynb` and `submissions/`), a feature-engineering iteration (`1.ipynb`, `data/`, and `submissions_1/`), and a same-day follow-up in `1.ipynb` that returns to a smaller feature set after the expanded set did not improve the 221-tree holdout RMSE. It is not a packaged library or a deployed service.

## What this demonstrates

- A cross-season setup that trains on 2015 flood labels and scores 2019 by aligning weekly precipitation, coordinates, elevation, and land cover onto one schema.
- A like-for-like comparison of nine regressors under 10-fold cross-validation, followed by submission files for the boosted models.
- Feature engineering aimed at how flooding accumulates: multi-week rainfall totals, whether rain was still rising, when the peak fell, and how elevation and land cover interact with that rain.
- Outlier filtering, repeated holdouts, importance ranking, and one-feature-at-a-time checks, with each feature set's RMSE left in the notebook so a larger set can be dropped when it does not improve the score.
- Prediction files that match the required columns, with outputs clipped to the 0–1 range of the training label.
