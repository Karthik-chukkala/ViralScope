# Module 4: Local Inference Demo

## Status

Implemented as a FastAPI service and responsive browser UI. The service loads the saved baseline models once per process, reports held-out comparisons, accepts planned-post inputs, and returns model-specific feature contributions. The demo is intended for local project review, not public deployment.

## Purpose and Boundaries

The demo turns a draft title, tags, category/context, and planned UTC publication time into a prediction of the historical SMP popularity label. It does not retrieve platform data, publish posts, access accounts, accept images, or predict current/live reach. No dataset images or user profiles are served.

The selected model can be TF-IDF + Ridge, context CatBoost, or text + context Ridge. The response includes the selected model, predicted score, planned time normalized to UTC, and up to six strongest signed feature contributions. These are model associations, not causes or guaranteed outcomes.

## API

- `GET /`: review UI.
- `GET /api/health`: local process health.
- `GET /api/dataset`: full-train label summary/histogram plus sample-only category, missing-title, text-length, and temporal aggregates.
- `GET /api/metrics`: chronological metrics/diagnostics plus aggregate user-disjoint metrics from `artifacts/baseline-metrics.json` and `artifacts/user-aware-metrics.json`, when present.
- The same metrics response includes the 15/20/25% temporal-sensitivity summary from `artifacts/temporal-sensitivity.json`, when present.
- `GET /api/diagnostics/{filename}`: one of the allowlisted sample EDA or holdout diagnostic PNGs.
- `POST /api/predict`: JSON input with `title`, `tags`, `category`, optional `subcategory` and `concept`, ISO `planned_at`, and model name.

The service expects trained files under `artifacts/models/`; it returns HTTP 503 with a clear message if they are missing. Pydantic bounds text lengths and model names. Naive input datetimes are interpreted as UTC; the UI labels the planned time as UTC and submits an explicit UTC offset.

## Run

From the project root, after caching a sample and training models:

```bash
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default and is not exposed to the network. The page shows the full-train target distribution, clearly labeled sampled category/text/time analysis, chronological, unseen-user, and temporal-sensitivity model comparisons, holdout diagnostic plots, and a synthetic random-example control that fills draft fields and selects a model. Users can edit the example before requesting an estimate/explanation.

## Validation

The API was exercised in-process against all three saved models; each returned HTTP 200, a finite prediction, and contribution entries. Metrics return both the 8,546-row chronological holdout and 9,050-row user-disjoint holdout, with zero user overlap. The browser displays the protocols separately. Unit tests cover UTC normalization, feature mapping, and that aggregate user split IDs are not returned. Run the full suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## License and Limitations

The page visibly displays **Powered by SMP Challenge** and the repository includes the required attribution in [Notice](../../Notice) and a copy of the Community License in [LICENSE-SMP-CHALLENGE](../../LICENSE-SMP-CHALLENGE.md). The seven-shard sample and baseline models are included in this research repository under that license. The incorporated AUP still requires review before reuse; images are not included or displayed.

Predictions inherit the constraints of a seven-shard training sample. The chronological holdout may share users across partitions; the separate group holdout is user-disjoint but not time-ordered. They are historical Flickr benchmark estimates and should not be interpreted as causal claims, user judgments, or platform-wide forecasts. Images remain deferred pending license review and a separate model milestone.