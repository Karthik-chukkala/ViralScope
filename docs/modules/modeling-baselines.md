# Module 3: Text and Context Modeling Baselines

## Status

Six learned variants are now fitted on all 305,613 labeled train rows from all 49 pinned shards. Holdout metrics remain separate: a chronological 20% post split, a user-disjoint 20% user-group split, and 15/20/25% chronological sensitivity windows. The public test split receives inference only because its labels are withheld. Per-prediction linear contributions and CatBoost SHAP values are exposed in the local demo.

## Purpose and Scope

Compare a simple constant predictor with text-only, structured-context, and combined text/context regressors. The model predicts the supplied SMP popularity `label` using fields available at a planned publication time. It does not use images, user/profile snapshots, engagement counts, IDs, coordinates, or the unlabeled public test split.

The source papers motivate comparing modalities and feature groups, but the selected-shard data and local chronological split differ from the official challenge protocol. Do not compare these scores directly with the reported 2024 paper table.

## Input and Features

The current full-train input is `data/train-all-safe.parquet`, with provenance in `data/train-all-safe-manifest.json`. It contains `Postdate`, `Category`, `Subcategory`, `Concept`, `Title`, `Alltags`, and `label` from all 49 train shards; it has no user or post IDs, profile fields, coordinates, or image paths. The original 42,730-row sample remains available as a fast smoke-test cache.

- **Text:** title and tags joined into one field, TF-IDF unigram/bigram terms, `min_df=2`, up to 150,000 features, and Ridge regression (`alpha=2`).
- **Text SGD:** the same TF-IDF representation with squared-error SGD and Huber-loss SGD (L2, `alpha=1e-5`, early stopping, seed 42).
- **Context:** category, subcategory, concept, and cyclic UTC hour, weekday, and month features, fitted with CatBoost (500 iterations, depth 8, learning rate 0.05).
- **Context shallow:** same context features with CatBoost (700 iterations, depth 5, learning rate 0.04).
- **Text + context:** the TF-IDF representation plus one-hot category/subcategory/concept and cyclic time features, fitted with Ridge (`alpha=2`).
- **Median:** the training-label median, as a non-learned reference.

The timestamp is parsed as Unix seconds. Invalid timestamps and missing, nonfinite, or negative labels are excluded before the split. No target is recomputed.

## Evaluation Protocol

Rows are ordered by `Postdate`; the final 20% by time is a local holdout and all earlier rows are training data. Equal timestamps at the cutoff go to the holdout. This future-time evaluation may share users across partitions. A separate user-disjoint group holdout is reported below. The current model fit uses all 49 physical train shards.

Report **SRC** (Spearman rank correlation) and **MAE** (challenge-aligned metrics), plus **RMSE** and **R²**. Training metrics diagnose fit; holdout metrics are the main result. The public HF test labels are withheld placeholders and are not scored.

## Completed Results

The full-train chronological run used 305,613 valid rows: 244,490 train and 61,123 future-time holdout, with cutoff `2016-01-17T11:26:49Z`.

| Model | Train SRC | Train MAE | Holdout SRC | Holdout MAE | Holdout RMSE | Holdout R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median | n/a | 1.954 | n/a | 1.990 | 2.488 | -0.008 |
| TF-IDF + Ridge | 0.924 | 0.609 | 0.612 | 1.508 | 1.951 | 0.381 |
| TF-IDF + SGD | 0.808 | 1.080 | 0.572 | 1.596 | 2.027 | 0.331 |
| TF-IDF + Huber SGD | 0.427 | 1.804 | 0.336 | 1.925 | 2.427 | 0.041 |
| Context CatBoost | 0.715 | 1.266 | 0.341 | 1.855 | 2.328 | 0.117 |
| Context CatBoost shallow | 0.649 | 1.420 | 0.337 | 1.853 | 2.335 | 0.112 |
| Text + context Ridge | 0.930 | 0.587 | 0.634 | 1.490 | 1.917 | 0.402 |

Text + context Ridge has the best full-train chronological holdout result (SRC 0.634, MAE 1.490). Huber SGD is the weakest variant here (SRC 0.336, MAE 1.925) and should be treated as a loss-function comparison, not a candidate winner. Train/holdout gaps remain large, so the scores are exploratory and no tuning was performed on the holdout.

### Holdout error diagnostics

The training run saves actual-versus-predicted, residual histogram, and per-category MAE charts under `artifacts/diagnostics/`, and adds residual summaries/category metrics to `artifacts/baseline-metrics.json`.

| Model | Mean error (predicted - actual) | Within 2 label units |
| --- | ---: | ---: |
| TF-IDF + Ridge | +0.438 | 71.1% |
| TF-IDF + SGD | +0.305 | 69.1% |
| TF-IDF + Huber SGD | -0.598 | 59.6% |
| Context CatBoost | +0.069 | 61.0% |
| Context CatBoost shallow | -0.044 | 60.8% |
| Text + context Ridge | +0.412 | 71.1% |

Positive mean error denotes average overprediction. Category metrics include groups with at least 50 holdout rows. They are descriptive subgroup diagnostics, not causal findings; per-category counts and six-model values are in `artifacts/baseline-metrics.json`.

The saved charts are `holdout-actual-vs-predicted.png`, `holdout-residuals.png`, and `holdout-mae-by-category.png`. The local UI renders all three through the allowlisted `/api/diagnostics/{filename}` route.

## User-Disjoint Evaluation

`viralscope.user_aware_eval` fetches only `Uid` from the exact 49 shards listed in the full-train manifest. It verifies the pinned revision and row alignment, uses IDs only as `GroupShuffleSplit` groups, and keeps ID values in memory. IDs are neither model features nor written to the cache, report, or repository. The fixed random seed is 42.

The full-train user-disjoint result uses 238,630 train posts from 30,649 users and 66,983 holdout posts from 7,663 unseen users; measured user overlap is zero.

| Model | Chronological SRC | Chronological MAE | Unseen-user SRC | Unseen-user MAE |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Ridge | 0.612 | 1.508 | 0.484 | 1.748 |
| TF-IDF + SGD | 0.572 | 1.596 | 0.459 | 1.705 |
| TF-IDF + Huber SGD | 0.336 | 1.925 | 0.209 | 1.868 |
| Context CatBoost | 0.341 | 1.855 | 0.258 | 1.929 |
| Context CatBoost shallow | 0.337 | 1.853 | 0.272 | 1.894 |
| Text + context Ridge | 0.634 | 1.490 | 0.546 | 1.708 |

The group holdout is randomly assigned by user and is not time-ordered. It complements, rather than replaces, the chronological evaluation. Text + context leads on the user-disjoint holdout; its performance is lower than on the chronological holdout, which is consistent with a harder unseen-user setting but does not isolate user effects from the different holdout composition. A future-time, unseen-user evaluation would require a combined split protocol.

The aggregate report is `artifacts/user-aware-metrics.json`; model files and plots are kept in `artifacts/models/user-aware/` and `artifacts/diagnostics/user-aware/`. The report contains user counts and overlap only, never raw IDs.

## Final Fit and Test Inference

After selecting/evaluating the model family, `viralscope.final_models` refits all six learned variants on every valid labeled train row. The final inference models are stored under `artifacts/models/final/`, with provenance in `artifacts/final-models-report.json`. These full-fit models do not have holdout metrics; use `artifacts/baseline-metrics.json` and `artifacts/user-aware-metrics.json` for evaluation.

The separate `viralscope.inference_cache` reads the six non-image test input columns from all 29 pinned test shards and verifies 180,581 rows against split metadata. It does not read/copy labels, IDs, images, profile data, or coordinates. `viralscope.test_inference` runs all six final models on all 180,581 rows and writes anonymous row-indexed prediction columns to `artifacts/test-inference-predictions.parquet`; aggregate prediction ranges are in `artifacts/test-inference-report.json`. Test MAE/SRC are intentionally not computed because the public labels are withheld as `-1`.

For interpretation only, the verified train label range is 1.00-16.56. Test predictions outside that range are not clipped: TF-IDF Ridge has 17, TF-IDF SGD has 1, Huber SGD has 0, context CatBoost has 0, shallow context CatBoost has 0, and text + context Ridge has 7 such predictions. These counts describe model output behavior and are not test errors.

## Temporal Sensitivity

`viralscope.temporal_sensitivity` evaluates chronological tail holdouts of 15%, 20%, and 25% using all pinned train shards and the same model settings. It writes one aggregate report to `artifacts/temporal-sensitivity.json`; temporary models and plots are deleted after each run.

| Holdout tail | Rows | Cutoff (UTC) | TF-IDF SRC / MAE | SGD SRC / MAE | Huber SRC / MAE | Context SRC / MAE | Shallow context SRC / MAE | Text + context SRC / MAE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15% | 45,842 | 2016-01-31 | 0.619 / 1.482 | 0.573 / 1.571 | 0.336 / 1.910 | 0.371 / 1.823 | 0.374 / 1.807 | 0.642 / 1.467 |
| 20% | 61,123 | 2016-01-17 | 0.612 / 1.508 | 0.572 / 1.596 | 0.336 / 1.925 | 0.341 / 1.855 | 0.337 / 1.853 | 0.634 / 1.490 |
| 25% | 76,404 | 2016-01-04 | 0.592 / 1.544 | 0.551 / 1.627 | 0.311 / 1.922 | 0.323 / 1.888 | 0.314 / 1.873 | 0.618 / 1.522 |

The holdouts are nested tails from all 49 train shards and overlap; they are sensitivity checks, not independent test sets. Text + context Ridge leads SRC and MAE at all three full-train windows. The separate seven-shard sensitivity results are preserved only in the Git history, not presented as current model results.

The complete report is `artifacts/baseline-metrics.json`. Models are stored under `artifacts/models/` and included with the research repository under the SMP Challenge Community License. Scores are not official test results and do not establish expected performance on the entire training split.

## Run and Reproduce

Build the all-train input cache, evaluate, refit final models on all labeled training rows, and run inference on all unlabeled test rows:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.sample_cache --sample-shards 49 --workers 4 --output data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json
.venv/bin/python -u -m viralscope.sample_eda --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --output artifacts/full-train-eda.json --plot artifacts/diagnostics/full-train-temporal-text.png
.venv/bin/python -u -m viralscope.train_baselines --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.user_aware_eval --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
.venv/bin/python -u -m viralscope.temporal_sensitivity --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
.venv/bin/python -u -m viralscope.final_models --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.inference_cache --workers 4
.venv/bin/python -u -m viralscope.test_inference
```

The cache and trained models are local artifacts; the commands do not download image bytes. Run tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Limitations and Next Checks

- Tune hyperparameters only on a separate validation split; do not select a final model by repeatedly consulting holdout windows.
- Design a combined future-time/unseen-user protocol if that is the intended deployment claim.
- Validate contribution explanations across representative examples and inspect them for text artifacts before any broader release.
- Generate and inspect residual/category plots for further cutoffs if they are used for model selection.
- Consider richer text representations only if validation results justify the added cost and complexity.
- Keep image training deferred until the supervisor reviews the license conditions for image use and display.