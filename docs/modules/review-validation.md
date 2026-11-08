# Review Validation: Combined Evaluation, Data Quality, and Explanations

## Combined future-time and unseen-user evaluation

Run the combined evaluation from the project root:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.combined_eval --workers 4
```

The script reads `Uid` from the exact pinned train shards listed in the all-train manifest, verifies row alignment with the safe cache, and keeps IDs in memory only. The chronological tail defines the future-time holdout. Every earlier row belonging to a user in that holdout is purged from training. Thus training rows precede the cutoff and the train/holdout user overlap is zero. The report and model artifacts contain aggregate metrics only.

The public test split remains inference-only and is never used for this evaluation.

Completed full-train result: 177,604 train posts, 61,123 future-time holdout posts, 66,886 purged pre-cutoff posts, 26,520 train users, 11,792 holdout users, and zero user overlap. The cutoff is `2016-01-17T11:26:49Z`. Text + context Ridge leads this holdout (SRC 0.515, MAE 1.733); these fixed configurations were not tuned on it.

## Full-train data-quality audit

```bash
.venv/bin/python -m viralscope.data_quality
```

The audit checks the all-train cache schema, missing and blank fields, timestamp validity/range, label numeric/finiteness/sign/range, category coverage, exact cached-row duplicates, repeated input rows, and repeated inputs with differing labels. The output is `artifacts/full-train-quality.json`; it contains aggregates and no raw text or IDs.

Duplicate counts are descriptive, not an instruction to drop records. The safe cache omits post IDs, and this check cannot establish identity between records whose cached fields differ. Conflicting labels for identical cached inputs are retained as a dataset finding rather than silently resolved.

Post identity was checked separately using the existing full-shard aggregate scan:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.metadata_eda --full --workers 4 --output artifacts/full-metadata-eda.json
```

The full scan found 305,613 unique post IDs among 305,613 rows (zero duplicate post IDs), 38,312 unique users, and no invalid timestamps. IDs are read only transiently; the report contains aggregate counts.

## Explanation fidelity and stability checks

```bash
.venv/bin/python -m viralscope.explanation_audit --rows 128
```

The audit applies the same shared contribution code used by the demo to a deterministic sample of the chronological train tail. It checks additive reconstruction (linear intercept or CatBoost SHAP base value plus all feature contributions), repeatability on identical inputs, and invariance to whitespace-only formatting changes. The aggregate output is `artifacts/explanation-stability.json`; no example text is written.

All six models passed on 128 rows each. Each explanation repeated identically, the maximum additive reconstruction error was below `7.2e-15`, the maximum prediction change under whitespace-only edits was zero, and top-six feature-set Jaccard / shared-feature sign agreement were both 1.0.

These are implementation and formatting-invariance checks, not proof that explanations are causal, robust to semantic paraphrases, or valid on a different population.
