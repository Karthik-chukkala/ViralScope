# ViralScope: Team Handoff

**As of:** 2026-11-08
**Current demo:** http://127.0.0.1:8000
**Project state:** A local dashboard/demo is running with six learned model variants fitted on all labeled training rows, full-train evaluation and EDA, plus inference over every unlabeled test row. Test accuracy is unavailable because test labels are withheld.

## 1. What We Are Building

ViralScope estimates the historical popularity score of a social post from information available before publication, then shows which model inputs contributed to its estimate. The current Part 1 scope is title, tags, category/context, and planned publication time. It does not predict current platform reach, query social platforms, publish posts, or use images.

The research basis is the SMP Challenge literature: the 2019 paper defines temporal popularity prediction and emphasizes Spearman rank correlation (SRC) and mean absolute error (MAE); the 2024 overview compares later challenge methods. The papers motivate our experiments but are not directly comparable to our local split protocol. We keep the dataset's supplied `label` rather than recomputing the paper's popularity equation, whose exact application in this release is not established.

## 2. Dataset Findings and Decisions

The source is Hugging Face `smpchallenge/SMP-Image`, pinned to revision `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`.

| Verified item | Finding |
| --- | --- |
| Full train rows | 305,613 across 49 shards / 3,080 row groups |
| Public test rows | 180,581; 381 sampled test labels were all `-1`, so this split is treated as unlabeled and never scored |
| Full-train label audit | 0 missing/non-numeric/non-finite labels; 0 `-1` sentinels; min 1.00, max 16.56, mean 6.4055, population SD 2.4730, median 6.23 |
| Full-train modeling cache | 305,613 rows from all 49 train shards; seven selected non-image columns, compressed Parquet about 15.9 MiB |
| Test inference cache | 180,581 rows from all 29 test shards; six non-image input columns, no labels or IDs, compressed Parquet about 8.4 MiB |
| Quick EDA sample | 42,730 rows from seven evenly spaced train shards; retained for fast smoke checks |
| Full-train text EDA | 10,553 missing/blank titles (3.45%), 3 missing/blank tag fields, no missing timestamps/categories, 11 categories |
| Full dataset size | About 37.78 GiB of original image-bearing Parquet; only column-projected non-image caches were created |
| Images | No dataset image bytes were included in train/test caches or the demo |

The full-train label mean and SD match the 2019 paper's rounded train statistics (6.41 and 2.47). This supports the target scale, but does not prove the exact label snapshot or formula. The HF statistics endpoint covers only 64,551 rows and is partial; it is not used as a full-train estimate.

### Why sampling was chosen

The first all-shard metadata scan was too slow. Remote range-request latency dominated the small selected-column payload: a serial shard read was about 57 seconds, while four parallel shards took about 118 seconds. Larger 4 MiB and 16 MiB blocks were slower than 64 KiB. The HF dataset `/rows` endpoint did not honor column projection and rejected requests above 100 rows. Local DuckDB/Polars would require downloading the 25.66 GB train Parquet split first, so that was not a useful shortcut.

The first resolution was a deterministic seven-shard smoke sample. The current modeling cache goes further: it reads only the needed columns from all 49 pinned train shards, with up to four concurrent readers and 64 KiB range blocks, while still avoiding image bytes. The smoke sample remains useful for quick checks; full-train metrics and EDA now use the all-shard projection. Both manifests list exact source shards and revision.

### License and data handling

The SMP Challenge Community License is research/non-commercial and includes attribution and acceptable-use conditions. `Notice` contains the required attribution text, and the website displays **Powered by SMP Challenge**. The complete current agreement/AUP still needs supervisor review before public deployment, distribution, or image use. Do not use the data or outputs to develop or enhance a foundational generative AI model.

The published train and test-input caches include raw title/tag text but exclude IDs, profile fields, coordinates, and image paths. Final models and anonymous row-indexed prediction columns are included as research/non-commercial derivatives under the SMP Challenge Community License, with the agreement and `Notice`. Reports contain no raw post text or user IDs. Complete papers, copied text extracts, and images are not included.

## 3. What Is Implemented

1. **Dataset access and audit:** pinned revision, license/provenance report, split sizes, and the complete label-only scan. Code: `viralscope/dataset_audit.py`, `viralscope/target_audit.py`.
2. **EDA:** quick sample checks plus full-train month/category/title/tag aggregates and charts. Code: `viralscope/metadata_eda.py`, `viralscope/sample_eda.py`; reports: `artifacts/metadata-eda.json`, `artifacts/full-train-eda.json`.
3. **Column-projected caches:** all train rows in `data/train-all-safe.parquet`; all test inputs in `data/test-inference-inputs.parquet`. Manifests record pinned revision/shards. No IDs, profiles, coordinates, or image bytes.
4. **Six learned model variants:** TF-IDF Ridge, squared-error SGD, Huber SGD, two CatBoost context models, and fused text/context Ridge, plus a median baseline. Chronological, user-disjoint, and multi-cutoff evaluation. Code: `viralscope/train_baselines.py`, `viralscope/user_aware_eval.py`, `viralscope/temporal_sensitivity.py`; final refit: `viralscope/final_models.py`.
5. **All-test inference:** all six final models generate predictions for 180,581 test rows. The public test split is unlabeled, so no test accuracy is reported. Code: `viralscope/inference_cache.py`, `viralscope/test_inference.py`; outputs: `artifacts/test-inference-predictions.parquet`, `artifacts/test-inference-report.json`.
6. **Presentation site:** Dashboard, Dataset, Models, Demo, and Solutions sections with full-train analyses, sample-vs-full provenance, three holdout protocols, six models, charts, random synthetic examples, and test prediction distributions. Code: `viralscope/demo_app.py`, `web/index.html`.
7. **Module documentation:** [Dataset Access and Audit](modules/dataset-access-and-audit.md), [Cleaning and EDA](modules/cleaning-and-eda.md), [Modeling Baselines](modules/modeling-baselines.md), and [Local Demo](modules/local-demo.md).

The local site is currently running on loopback at port 8000. It was checked in a browser at desktop and mobile widths; form submission, dataset/text/time visualizations, both evaluation panels, holdout diagnostic plots, and explanations work without horizontal overflow.

## 4. Model Results

The principal evaluation uses all 305,613 labeled train rows: 244,490 for training and 61,123 for a chronological future-time holdout. The cutoff is `2016-01-17T11:26:49Z`. These are local holdout results, not official challenge-test results.

| Model | Train SRC | Train MAE | Holdout SRC | Holdout MAE | Holdout RMSE | Holdout R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median | n/a | 1.954 | n/a | 1.990 | 2.488 | -0.008 |
| TF-IDF + Ridge | 0.924 | 0.609 | 0.612 | 1.508 | 1.951 | 0.381 |
| TF-IDF + SGD | 0.808 | 1.080 | 0.572 | 1.596 | 2.027 | 0.331 |
| TF-IDF + Huber SGD | 0.427 | 1.804 | 0.336 | 1.925 | 2.427 | 0.041 |
| Context CatBoost | 0.715 | 1.266 | 0.341 | 1.855 | 2.328 | 0.117 |
| Context CatBoost shallow | 0.649 | 1.420 | 0.337 | 1.853 | 2.335 | 0.112 |
| Text + context Ridge | 0.930 | 0.587 | 0.634 | 1.490 | 1.917 | 0.402 |

Text + context Ridge leads the chronological holdout (SRC 0.634, MAE 1.490). Huber SGD is the weakest variant and remains as a loss-function comparison. Train/holdout gaps still indicate overfitting; parameters were not tuned on the holdout.

Full-train holdout residual, actual/predicted, and category-error plots are saved under `artifacts/diagnostics/`. Category metrics exclude groups with fewer than 50 holdout rows.

The chronological split is **not user-independent**. A separate random user-group holdout now has zero user overlap, but is not time-ordered. A combined future-time/unseen-user protocol remains a limitation.

The complementary full-train user-disjoint split has 238,630 train posts from 30,649 users and 66,983 holdout posts from 7,663 unseen users, with zero user overlap. The 20% split fraction applies to user groups, so the resulting holdout row share is approximate.

| Model | Unseen-user SRC | Unseen-user MAE | Unseen-user RMSE | Unseen-user R² |
| --- | ---: | ---: | ---: | ---: |
| Median | n/a | 1.895 | 2.439 | -0.002 |
| TF-IDF + Ridge | 0.484 | 1.748 | 2.224 | 0.167 |
| TF-IDF + SGD | 0.459 | 1.705 | 2.154 | 0.218 |
| TF-IDF + Huber SGD | 0.209 | 1.868 | 2.404 | 0.026 |
| Context CatBoost | 0.258 | 1.929 | 2.440 | -0.003 |
| Context CatBoost shallow | 0.272 | 1.894 | 2.403 | 0.027 |
| Text + context Ridge | 0.546 | 1.708 | 2.159 | 0.215 |

The fused model leads on this unseen-user holdout, while all models have weaker text SRC/MAE than on the chronological holdout. This is a different split population, not proof that user novelty alone caused the change. The report is `artifacts/user-aware-metrics.json`; it contains aggregate user counts/overlap only, not raw user IDs.

Temporal sensitivity was checked at 15%, 20%, and 25% holdout tails on all 305,613 train rows. Text + context Ridge leads SRC and MAE at each cutoff (15%: 0.642/1.467; 20%: 0.634/1.490; 25%: 0.618/1.522). These nested windows overlap and are sensitivity checks, not independent test sets. The aggregate report is `artifacts/temporal-sensitivity.json`; temporary model artifacts are discarded after the run.

Final inference models were refit on all 305,613 labeled train rows. All 180,581 public test rows were run through the six final models; the result is prediction distributions only because test labels are withheld. Anonymous predictions are saved in `artifacts/test-inference-predictions.parquet`, with ranges and quantiles in `artifacts/test-inference-report.json`. Test MAE/SRC are intentionally not computed.

The train-label range is 1.00-16.56. Unclipped test outputs outside that range occur 17 times for TF-IDF Ridge, once for TF-IDF SGD, and 7 times for text + context Ridge; neither CatBoost model nor Huber SGD produced out-of-range values. These are output sanity counts, not errors against test targets.

## 5. Run It

Activate or use the existing project environment. The current setup is WSL Ubuntu 24.04, Python 3.12, with an RTX 4050; these baselines run on CPU and need no GPU.

```bash
# Cache the seven safe model columns from all 49 pinned train shards
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.sample_cache --sample-shards 49 --workers 4 --output data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json

# Generate full-train EDA and charts
.venv/bin/python -u -m viralscope.sample_eda --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --output artifacts/full-train-eda.json --plot artifacts/diagnostics/full-train-temporal-text.png

# Evaluate six variants on a chronological holdout
.venv/bin/python -u -m viralscope.train_baselines --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4

# Evaluate a disjoint-user holdout; IDs are used in memory only
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.user_aware_eval --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4

# Compare cutoffs on full train, refit final models on all train rows, and infer on all test rows
.venv/bin/python -u -m viralscope.temporal_sensitivity --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
.venv/bin/python -u -m viralscope.final_models --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.inference_cache --workers 4
.venv/bin/python -u -m viralscope.test_inference

# Run unit tests
.venv/bin/python -m unittest discover -s tests -v

# Serve the local review app (already running at port 8000 in the current workspace)
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Hugging Face commands require the local credential. `HF_HUB_DOWNLOAD_TIMEOUT=120` avoids the client's default 10-second timeout on shard reads. Never put credentials in the repository or paste them into logs.

The API routes are `GET /api/health`, `GET /api/dataset`, `GET /api/metrics`, `GET /api/diagnostics/{filename}`, and `POST /api/predict`. Prediction input is title, tags, category, optional subcategory/concept, planned UTC datetime, and model name (`tfidf_ridge`, `tfidf_sgd`, `tfidf_sgd_huber`, `context_catboost`, `context_catboost_shallow`, or `text_context_ridge`).

Latest test result: **27 tests passed** after full-train fitting and test inference. API smoke checks returned HTTP 200 for all six final models, full-train EDA, all-test prediction summaries, and all three evaluation views.

## 6. What Remains

Prioritize work in this order:

1. **Strengthen evaluation:** consider a combined future-time/unseen-user protocol and keep model tuning on a separate validation split. The existing user-disjoint check is a random user-group split, not a future-time test; the three temporal sensitivity windows are nested, not independent.
2. **Broaden EDA:** add more type/range and duplicate policy checks, plus interpretation of full-train distributions. Keep the 42,730-row smoke findings labeled as sample-only; image quality remains deferred until license review.
3. **Validate explanations:** inspect linear term contributions and CatBoost SHAP values on representative inputs; test explanation stability and avoid causal wording.
4. **Reproducibility:** record/pin the successful dependency versions and add a concise offline review procedure with a tested sample input.
5. **License gate:** have the supervisor review the full Community License and AUP before any public deployment or redistribution.
6. **Image branch:** only after license approval and after the text/context milestone is accepted, consider a frozen pretrained image feature extractor and cached embeddings. The current caches intentionally omit images.

Do not report sample EDA as full-train metadata statistics, call the public `-1` test split labeled, claim official test performance, or imply these historical Flickr scores forecast current engagement on another platform.