# ViralScope

ViralScope is a research prototype for estimating the historical SMP Challenge popularity label of a draft social post before publication. Part 1 uses text and planned-time/category context; image modeling is deferred. This is not a forecast of live engagement on another platform.

## Current Demo

The local FastAPI demo serves full-train dataset analysis, separate sample-only charts, six model variants, chronological and user-disjoint holdout metrics, temporal cutoff sensitivity, test-split prediction distributions (not scores), diagnostic plots, and an interactive prediction form with feature contributions and random synthetic examples. Start it after generating the local data/model artifacts:

```bash
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The demo is intended for local review and binds to loopback by default.

## Reproduce

Use the project environment and a local Hugging Face login for gated dataset access. These commands project only non-image text/context/label columns from all 49 train shards and non-image inputs from all 29 test shards. They do not download image bytes, user IDs, profile fields, or coordinates.

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.sample_cache --sample-shards 49 --workers 4 --output data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json
.venv/bin/python -u -m viralscope.sample_eda --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --output artifacts/full-train-eda.json --plot artifacts/diagnostics/full-train-temporal-text.png
.venv/bin/python -u -m viralscope.train_baselines --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.user_aware_eval --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
.venv/bin/python -u -m viralscope.temporal_sensitivity --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
.venv/bin/python -u -m viralscope.final_models --data data/train-all-safe.parquet --manifest data/train-all-safe-manifest.json --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.inference_cache --workers 4
.venv/bin/python -u -m viralscope.test_inference
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

This research repository includes a 42,730-row smoke-test sample, a 305,613-row all-train cache, and a 180,581-row unlabeled test-input cache, with manifests, six final models, evaluation/EDA reports, and anonymous test predictions. These are SMP Challenge materials or derivatives distributed for research/non-commercial use under [the SMP Challenge Community License](LICENSE-SMP-CHALLENGE.md); retain `Notice` and review the AUP before reuse. The caches include title/tag text but no user/post IDs, image bytes, coordinates, or profile fields. Do not use these materials for commercial purposes or foundational generative AI development.

The full image-bearing dataset, image assets, and copied research papers/text extracts are not included. Research paper references and source links are listed in the [research blueprint](docs/research-and-project-blueprint.md).

## Project Notes

- [Team handoff](docs/team-handoff.md): end-to-end status, results, commands, limitations, and next steps.
- [Research blueprint](docs/research-and-project-blueprint.md)
- [Dataset access and audit](docs/modules/dataset-access-and-audit.md)
- [Cleaning and EDA](docs/modules/cleaning-and-eda.md)
- [Model baselines](docs/modules/modeling-baselines.md)
- [Local demo](docs/modules/local-demo.md)

The selected dataset revision is `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`. The public test split is treated as unlabeled because sampled labels were `-1`; predictions are generated, but no test scores are reported. Evaluation metrics come from local full-train chronological and user-disjoint holdouts, not official test results.

## License

The dataset is governed by the SMP Challenge Community License and an incorporated Acceptable Use Policy. A copy of the March 17, 2025 agreement is included; `Notice` contains the required attribution text, and the demo displays **Powered by SMP Challenge**. The AUP is incorporated by reference and must be reviewed before reuse. Image use/display remains out of scope.