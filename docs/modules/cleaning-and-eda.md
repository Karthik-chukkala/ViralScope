# Module 2: Cleaning and Exploratory Data Analysis

## Status

**Full pinned-train target-label and text/time EDA: complete.** The project now caches the seven required non-image modeling columns from all 49 pinned train shards. Full-train month/category/title/tag summaries are saved without raw text or IDs. Image-quality analysis remains deferred; modeling results are documented in [Modeling Baselines](modeling-baselines.md).

## Purpose

Establish trustworthy data-quality and exploratory findings before feature engineering or model training. Keep the benchmark's supplied label intact, identify withheld-label sentinels, and make the analysis reproducible without loading the image corpus into RAM.

## Input and Scope

- Repository: `smpchallenge/SMP-Image`.
- Revision: `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`.
- Split for supervised analysis: labeled `train` only.
- The public `test` split is not used for supervised metrics: sampled labels were `-1.0` and are treated as withheld placeholders pending confirmation by the dataset owners.
- Images are not read by the target audit. The train Parquet corpus is about 25.66 GB, but its label column can be read using Parquet range requests.

The audit compares the pinned repository SHA with the current main SHA before using the viewer size endpoint. Since that endpoint does not expose a revision, the code checks the split label count against the viewer's train row count and refuses to write a report if they disagree.

## Metadata EDA Sample

`viralscope/metadata_eda.py` defaults to seven evenly spaced train shards and four concurrent readers. For the pinned 49-shard release this selects shard indexes 0, 8, 16, 24, 32, 40, and 48. The report records the selected shard names, number scanned, available shard count, row count, revision, and that the scan is sampled. This deterministic coverage across the shard order is a fast exploratory slice, not a statistically random sample and not evidence for full-train frequencies.

The EDA reader projects only `Uid`, `Pid`, `Postdate`, `Category`, `Subcategory`, `Concept`, `Title`, and `Alltags`; it does not read image bytes, coordinates, or profile fields. It writes aggregate counts only, not raw text or IDs. Model training will use a separate ignored local sample cache that includes the supplied `label` and only prediction-time-safe fields; no image model is required for this first milestone.

Run the bounded sample:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.metadata_eda
```

To adjust sampling/concurrency, use `--sample-shards 7 --workers 4`. To restore a complete metadata scan, use `--full --workers 4`; expect it to take substantially longer because remote range-request latency dominates the small selected-column payload. `HF_HUB_DOWNLOAD_TIMEOUT=120` raises the Hugging Face client's 10-second default, which was too short for some shard reads.

### Completed seven-shard sample findings

The pinned sample contains 42,730 rows from seven evenly spaced shards. It has 10,976 distinct users, no duplicate post IDs in this sample, no missing timestamps/categories/tags, and 1,474 missing titles (3.45%). The most frequent categories are Travel&Active&Sports (9,047), Holiday&Celebrations (7,614), and Fashion (6,381). These figures describe the selected shards only; do not extrapolate them as exact full-train rates. The aggregate output is `artifacts/metadata-eda.json`.

The seven-shard smoke-test sample is `data/train-sample.parquet`, with source revision and shard list in `data/train-sample-manifest.json`. The full training cache is `data/train-all-safe.parquet`, with provenance in `data/train-all-safe-manifest.json`. Both contain `label`, `Postdate`, `Category`, `Subcategory`, `Concept`, `Title`, and `Alltags`, and exclude IDs, profile/location fields, and image paths. The full cache contains raw title/tag text; distribution is subject to the SMP Challenge Community License and its use restrictions.

### Text and temporal profile

`viralscope.sample_eda` summarizes the seven-shard smoke-test sample and the all-shard train cache without another network read. The sample-only report remains at `artifacts/sample-eda.json` and should not be treated as full-train EDA.

### Completed full-train text/time findings

The all-shard report `artifacts/full-train-eda.json` covers 305,613 train rows from all 49 shards. It found 10,553 missing/blank titles (3.45%), three missing/blank tag fields, no missing timestamps/categories, and no invalid timestamps. Dates cover `2015-02-28T16:00:00Z` through `2016-02-29T15:54:26Z`. Nonempty titles have a 19-character median and 3-word median; tag fields have a 16-token median. Categories represented: 11. The sample label mean is 6.4055, matching the complete label audit.

The full-train month/category/title/tag charts are saved in `artifacts/diagnostics/full-train-temporal-text.png`. The report contains aggregate counts and quantiles only, no raw text or IDs. The separate seven-shard chart remains `artifacts/diagnostics/sample-temporal-text.png` and is explicitly sample-only.

Run the local EDA without network access after the sample cache exists:

```bash
.venv/bin/python -u -m viralscope.sample_eda
```

For all train rows after building the full cache:

```bash
.venv/bin/python -u -m viralscope.sample_eda --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --output artifacts/full-train-eda.json --plot artifacts/diagnostics/full-train-temporal-text.png
```

## Implemented Slice: Target Labels

`viralscope/target_audit.py` reads each `data/train-*.parquet` shard from Hugging Face with `HfFileSystem`, then reads the `label` column only for that shard using PyArrow. It does not read title, tags, user/profile fields, image metadata, or image bytes. Labels are accumulated as scalar values (about 306K floats), not as decoded image objects.

The report includes:

- Number of rows scanned, shard count, and row-group count.
- Compressed label-column size from Parquet metadata.
- Missing, nonnumeric, nonfinite, exact `-1` sentinel, and other negative value counts.
- Summary statistics with population standard deviation and selected quantiles, both including finite labels and excluding the exact `-1` sentinel.
- A 15-bin histogram of non-sentinel labels.

The script does not change or recompute target labels. The 2019 paper's equation is $s = \log_2(r/d) + 1$, but whether the HF release applies that exact formula and snapshot is not established; provided labels remain authoritative.

### Completed full-train results

Run against the pinned revision with a 120-second Hugging Face read timeout:

| Check | Result |
| --- | ---: |
| Rows scanned | 305,613 |
| Train shards / row groups | 49 / 3,080 |
| Missing / nonnumeric / nonfinite values | 0 / 0 / 0 |
| `-1` sentinels / other negative labels | 0 / 0 |
| Minimum / maximum | 1.00 / 16.56 |
| Mean / population SD | 6.4055 / 2.4730 |
| Median | 6.23 |

The mean and standard deviation match the 2019 paper's train values (6.41 and 2.47) after rounding. This supports that the HF training labels are on the same target scale, though it does not prove the exact view snapshot or formula used. The result is saved in `artifacts/train-target-audit.json`. The separate HF statistics endpoint is marked partial (64,551 rows) and should not be used as full-train statistics.

## Run and Outputs

From the project root:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.target_audit
```

Default output: `artifacts/train-target-audit.json`. To change the output path or revision:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.target_audit --revision f131dba653a05c0e3c6294388f81ed3fbd9c71a0 --output artifacts/train-target-audit.json
```

The command requires the local Hugging Face credential, prints no credential or row content, and fails before writing output if the revision changed, label column is missing, or scanned train-row count disagrees with viewer metadata. Use `HF_HUB_DOWNLOAD_TIMEOUT=120` because the client's 10-second default can time out on occasional shard reads.

Focused tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Cleaning and EDA Work Still Planned

The next work within this module will add independently validated summaries for:

1. Missingness and type/range validation for selected metadata fields.
2. Duplicate `Pid`/rows and overlap diagnostics, without dropping records until the policy is justified.
3. Category/subcategory/concept frequencies and imbalance.
4. Title and tag coverage, lengths, and common-term analysis with privacy-conscious output.
5. Postdate parsing, chronology, and train-only time coverage.
6. User/profile field availability and prediction-time leakage review.
7. Bounded image decode/missing/corrupt checks and image dimension summaries.
8. EDA plots and a written interpretation for each result.

These summaries must be computed from the pinned train split or explicitly labeled as partial. The HF statistics endpoint is marked `partial: true` and covers only 64,551 rows; its results are useful for smoke checks, not as full-train EDA. The seven-shard metadata output is also sample-only and must not be described as full-train EDA.

## Validation and Completion

The target-label slice is complete and unit-tested. This module is complete when the remaining full-train quality checks and modality summaries are reproducible, each cleaning decision has before/after counts, EDA results are interpreted, and the feature leakage policy is explicit. No image or test split should be downloaded merely to produce train-label statistics.

## Privacy and License

Do not include raw user descriptions, tags from individual posts, coordinates, or image URLs in the report. Aggregate location data only. Keep the public image demo behind the SMP license/AUP review gate documented in [Dataset Access and Audit](dataset-access-and-audit.md).