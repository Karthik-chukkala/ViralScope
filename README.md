# ViralScope

ViralScope is a research prototype for estimating the historical SMP Challenge popularity label of a draft social post before publication. Part 1 uses text and planned-time/category context; image modeling is deferred. This is not a forecast of live engagement on another platform.

## Current Demo

The local FastAPI demo serves a dataset snapshot, sampled category/text/time analysis, chronological and user-disjoint holdout metrics, temporal cutoff sensitivity, residual/category diagnostic plots, and an interactive prediction form with feature contributions and random synthetic examples. Start it after generating the local data/model artifacts:

```bash
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The demo is intended for local review and binds to loopback by default.

## Reproduce

Use the project environment and a local Hugging Face login for gated dataset access. The commands below read only selected non-image columns for the seven-shard modeling sample; they do not download the full Parquet or image corpus.

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.metadata_eda
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.sample_cache --sample-shards 7 --workers 4
.venv/bin/python -u -m viralscope.sample_eda
.venv/bin/python -u -m viralscope.train_baselines --workers 4
HF_HUB_DOWNLOAD_TIMEOUT=120 .venv/bin/python -u -m viralscope.user_aware_eval --workers 4
.venv/bin/python -u -m viralscope.temporal_sensitivity --workers 4
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

This research repository includes the 42,730-row, seven-shard training sample, its manifest, trained baseline models, and aggregate audit reports. These are SMP Challenge materials or derivatives distributed for research/non-commercial use under [the SMP Challenge Community License](LICENSE-SMP-CHALLENGE.md); retain the `Notice` file and review the agreement/AUP before reuse. The sample contains post titles and tags, but no user IDs, post IDs, images, coordinates, or profile fields. Do not use these materials for commercial purposes or foundational generative AI development.

The full image-bearing dataset, image assets, and copied research papers/text extracts are not included. Research paper references and source links are listed in the [research blueprint](docs/research-and-project-blueprint.md).

## Project Notes

- [Team handoff](docs/team-handoff.md): end-to-end status, results, commands, limitations, and next steps.
- [Research blueprint](docs/research-and-project-blueprint.md)
- [Dataset access and audit](docs/modules/dataset-access-and-audit.md)
- [Cleaning and EDA](docs/modules/cleaning-and-eda.md)
- [Model baselines](docs/modules/modeling-baselines.md)
- [Local demo](docs/modules/local-demo.md)

The selected dataset revision is `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`. The public test split is treated as unlabeled because sampled labels were `-1`; do not score it. Baseline metrics are from seven sampled shards and local chronological/user-disjoint holdouts, not official test results.

## License

The dataset is governed by the SMP Challenge Community License and an incorporated Acceptable Use Policy. A copy of the March 17, 2025 agreement is included; `Notice` contains the required attribution text, and the demo displays **Powered by SMP Challenge**. The AUP is incorporated by reference and must be reviewed before reuse. Image use/display remains out of scope.