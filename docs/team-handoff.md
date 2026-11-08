# ViralScope: Team Handoff

**As of:** 2026-11-07
**Current demo:** http://127.0.0.1:8000
**Project state:** A local, end-to-end text/context baseline demo is running. Results are exploratory and based on a reproducible shard sample, not the full training split.

## 1. What We Are Building

ViralScope estimates the historical popularity score of a social post from information available before publication, then shows which model inputs contributed to its estimate. The current Part 1 scope is title, tags, category/context, and planned publication time. It does not predict current platform reach, query social platforms, publish posts, or use images.

The research basis is the SMP Challenge literature: the 2019 paper defines temporal popularity prediction and emphasizes Spearman rank correlation (SRC) and mean absolute error (MAE); the 2024 overview compares later challenge methods. The papers motivate our experiments but are not directly comparable to our sample/split. We keep the dataset's supplied `label` rather than recomputing the paper's popularity equation, whose exact application in this release is not established.

## 2. Dataset Findings and Decisions

The source is Hugging Face `smpchallenge/SMP-Image`, pinned to revision `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`.

| Verified item | Finding |
| --- | --- |
| Full train rows | 305,613 across 49 shards / 3,080 row groups |
| Public test rows | 180,581; 381 sampled test labels were all `-1`, so this split is treated as unlabeled and never scored |
| Full-train label audit | 0 missing/non-numeric/non-finite labels; 0 `-1` sentinels; min 1.00, max 16.56, mean 6.4055, population SD 2.4730, median 6.23 |
| Metadata sample | 42,730 rows from seven evenly spaced shards: 0, 8, 16, 24, 32, 40, 48 |
| Sample metadata | 10,976 unique users; 0 repeated post IDs in the sample; 1,474 missing titles (3.45%) |
| Full dataset size | About 37.78 GiB of Parquet; no full split was downloaded |
| Images | No dataset image bytes were included in the sample cache or demo |

The full-train label mean and SD match the 2019 paper's rounded train statistics (6.41 and 2.47). This supports the target scale, but does not prove the exact label snapshot or formula. The HF statistics endpoint covers only 64,551 rows and is partial; it is not used as a full-train estimate.

### Why sampling was chosen

The first all-shard metadata scan was too slow. Remote range-request latency dominated the small selected-column payload: a serial shard read was about 57 seconds, while four parallel shards took about 118 seconds. Larger 4 MiB and 16 MiB blocks were slower than 64 KiB. The HF dataset `/rows` endpoint did not honor column projection and rejected requests above 100 rows. Local DuckDB/Polars would require downloading the 25.66 GB train Parquet split first, so that was not a useful shortcut.

The resolution is a deterministic sample of seven shards spread across the pinned 49-shard train split, with up to four concurrent readers and 64 KiB range blocks. It is a fast exploratory slice, **not a random row sample or a guarantee of full-train frequencies**. The report lists the exact source shards and revision.

### License and data handling

The SMP Challenge Community License is research/non-commercial and includes attribution and acceptable-use conditions. `Notice` contains the required attribution text, and the website displays **Powered by SMP Challenge**. The complete current agreement/AUP still needs supervisor review before public deployment, distribution, or image use. Do not use the data or outputs to develop or enhance a foundational generative AI model.

The published training sample includes raw title/tag text but excludes IDs, profile fields, coordinates, and image paths. The sample and trained models are derivatives distributed for research/non-commercial use under the SMP Challenge Community License, with a copy of the agreement and `Notice` included. Aggregate reports contain no raw post text or user IDs. Complete papers, copied text extracts, and images are not included.

## 3. What Is Implemented

1. **Dataset access and audit:** pinned revision, license/provenance report, split sizes, and the complete label-only scan. Code: `viralscope/dataset_audit.py`, `viralscope/target_audit.py`.
2. **Sample metadata EDA:** aggregate missingness, ID uniqueness, categories, concepts, monthly counts, and title/tag lengths. Code: `viralscope/metadata_eda.py`; output: `artifacts/metadata-eda.json`.
3. **Local modeling cache:** selected non-image columns plus `label`, saved to `data/train-sample.parquet`; provenance is in `data/train-sample-manifest.json`. Code: `viralscope/sample_cache.py`.
4. **Baseline ladder:** median, TF-IDF + Ridge, context CatBoost, and text + context Ridge. Chronological and user-disjoint holdouts are both evaluated. Code: `viralscope/train_baselines.py`, `viralscope/user_aware_eval.py`; reports: `artifacts/baseline-metrics.json`, `artifacts/user-aware-metrics.json`; model files: `artifacts/models/`.
5. **Inference and review site:** FastAPI serves the UI, aggregate dataset snapshot/histogram, category sample chart, both holdout metric sets, residual/category diagnostics, and prediction endpoint. The form prefills a synthetic example and can shuffle to another draft/model without typing. Text models explain signed feature contributions; CatBoost uses SHAP values. Code: `viralscope/demo_app.py`, `web/index.html`.
6. **Module documentation:** [Dataset Access and Audit](modules/dataset-access-and-audit.md), [Cleaning and EDA](modules/cleaning-and-eda.md), [Modeling Baselines](modules/modeling-baselines.md), and [Local Demo](modules/local-demo.md).

The local site is currently running on loopback at port 8000. It was checked in a browser at desktop and mobile widths; form submission, dataset visualizations, holdout diagnostic plots, and explanations work without horizontal overflow.

## 4. Model Results

The evaluation uses all 42,730 sample rows with a chronological split: 34,184 train rows and 8,546 future-time holdout rows. The cutoff is `2016-01-11T08:05:51Z`. These are **local sample results**, not scores on the public test split or official challenge test.

| Model | Train SRC | Train MAE | Holdout SRC | Holdout MAE | Holdout RMSE | Holdout R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median | n/a | 2.066 | n/a | 2.013 | 2.494 | -0.000 |
| TF-IDF + Ridge | 0.938 | 0.696 | 0.575 | 1.623 | 2.058 | 0.319 |
| Context CatBoost | 0.788 | 1.152 | 0.317 | 1.898 | 2.383 | 0.087 |
| Text + context Ridge | 0.941 | 0.642 | 0.578 | 1.642 | 2.066 | 0.314 |

Text + context slightly improves SRC over text alone, but is slightly worse on MAE, RMSE, and R². Therefore, the current evidence does not show that fusion improves overall score estimation. Large train/holdout gaps indicate overfit, especially for sparse text features. Parameters were not tuned on the holdout.

The holdout diagnostic report shows mean prediction errors of +0.543 for TF-IDF Ridge, +0.131 for context CatBoost, and +0.537 for text + context Ridge (positive means overprediction). Within two label units, the models cover 68.4%, 59.1%, and 67.2% of holdout rows, respectively. Category MAE slices and actual/predicted and residual plots are saved under `artifacts/diagnostics/`; these remain sample-only diagnostics.

The chronological split is **not user-independent**. A separate random user-group holdout now has zero user overlap, but is not time-ordered. A combined future-time/unseen-user protocol and the seven-shard sampling method remain important limitations.

The complementary user-disjoint split has 33,680 train posts from 8,780 users and 9,050 holdout posts from 2,196 users, with zero user overlap. The 20% split fraction applies to user groups, so the resulting holdout row share is approximate.

| Model | Unseen-user SRC | Unseen-user MAE | Unseen-user RMSE | Unseen-user R² |
| --- | ---: | ---: | ---: | ---: |
| Median | n/a | 2.085 | 2.626 | -0.022 |
| TF-IDF + Ridge | 0.443 | 1.835 | 2.336 | 0.191 |
| Context CatBoost | 0.428 | 1.890 | 2.395 | 0.150 |
| Text + context Ridge | 0.548 | 1.775 | 2.223 | 0.268 |

The fused model leads on this unseen-user holdout, while all models have weaker text SRC/MAE than on the chronological holdout. This is a different split population, not proof that user novelty alone caused the change. The report is `artifacts/user-aware-metrics.json`; it contains aggregate user counts/overlap only, not raw user IDs.

## 5. Run It

Activate or use the existing project environment. The current setup is WSL Ubuntu 24.04, Python 3.12, with an RTX 4050; these baselines run on CPU and need no GPU.

```bash
# Aggregate the seven-shard metadata sample
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.metadata_eda

# Build/rebuild the local, image-free modeling cache
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.sample_cache --sample-shards 7 --workers 4

# Train models and refresh metrics
.venv/bin/python -u -m viralscope.train_baselines --workers 4

# Evaluate a disjoint user holdout; user IDs are read in memory and never saved
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.user_aware_eval --workers 4

# Run unit tests
.venv/bin/python -m unittest discover -s tests -v

# Serve the local review app (already running at port 8000 in the current workspace)
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Hugging Face commands require the local credential. `HF_HUB_DOWNLOAD_TIMEOUT=120` avoids the client's default 10-second timeout on shard reads. Never put credentials in the repository or paste them into logs.

The API routes are `GET /api/health`, `GET /api/dataset`, `GET /api/metrics`, `GET /api/diagnostics/{filename}`, and `POST /api/predict`. Prediction input is title, tags, category, optional subcategory/concept, planned UTC datetime, and model name (`tfidf_ridge`, `context_catboost`, or `text_context_ridge`).

Latest test result: **17 tests passed**. API smoke checks returned HTTP 200 for health, dataset summary, chronological/user-aware metrics, diagnostic plots, and predictions/explanations from all three saved models.

## 6. What Remains

Prioritize work in this order:

1. **Strengthen evaluation:** compare multiple temporal cutoffs and design a combined future-time/unseen-user protocol. The existing user-disjoint check is a random user-group split, not a future-time test.
2. **Broaden EDA:** add tag/title length and coverage analysis, fuller temporal summaries, missing/type/range checks, residual and per-category model error slices, and concise interpretations. Continue to label shard-derived metadata results as sample-only.
3. **Validate explanations:** inspect linear term contributions and CatBoost SHAP values on representative inputs; test explanation stability and avoid causal wording.
4. **Reproducibility:** record/pin the successful dependency versions and add a concise offline review procedure with a tested sample input.
5. **License gate:** have the supervisor review the full Community License and AUP before any public deployment or redistribution.
6. **Image branch:** only after license approval and after the text/context milestone is accepted, consider a frozen pretrained image feature extractor and cached embeddings. Do not download the full image-bearing split just to continue current text/context work.

Do not report sample EDA as full-train metadata statistics, call the public `-1` test split labeled, claim official test performance, or imply these historical Flickr scores forecast current engagement on another platform.