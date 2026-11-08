# Module 4: Local Inference Demo

## Status

Implemented as a FastAPI service and responsive browser UI. The service loads the saved baseline models once per process, reports held-out comparisons, accepts planned-post inputs, and returns model-specific feature contributions. The demo is intended for local project review, not public deployment.

## Purpose and Boundaries

The demo turns a draft title, tags, category/context, and planned UTC publication time into a prediction of the historical SMP popularity label. It does not retrieve platform data, publish posts, access accounts, accept images, or predict current/live reach. No dataset images or user profiles are served.

The selected model can be TF-IDF + Ridge, squared-error SGD, Huber SGD, standard or shallow context CatBoost, or text + context Ridge. These six final models are refit on all 305,613 labeled training rows. The response includes the selected model, predicted score, planned time normalized to UTC, and up to six strongest signed feature contributions. These are model associations, not causes or guaranteed outcomes.

## API

- `GET /`: project dashboard; `/dataset`, `/models`, `/demo`, `/solutions`, and `/references` provide dedicated views.
- `GET /api/health`: local process health.
- `GET /api/dataset`: full-train label/text/time/category summaries, separate sample-only charts, and test prediction distributions without test metrics.
- `GET /api/metrics`: chronological metrics/diagnostics plus aggregate user-disjoint metrics from `artifacts/baseline-metrics.json` and `artifacts/user-aware-metrics.json`, when present.
- The same metrics response includes the 15/20/25% temporal-sensitivity summary from `artifacts/temporal-sensitivity.json`, when present.
- `GET /api/diagnostics/{filename}`: one of the allowlisted sample EDA or holdout diagnostic PNGs.
- `POST /api/predict`: JSON input with `title`, `tags`, `category`, optional `subcategory` and `concept`, ISO `planned_at`, and model name.

The service expects final full-train model files under `artifacts/models/final/`; it returns HTTP 503 with a clear message if they are missing. Pydantic bounds text lengths and model names. Naive input datetimes are interpreted as UTC; the UI labels the planned time as UTC and submits an explicit UTC offset.

## Run

From the project root, after building the full-train cache, evaluating the holdouts, fitting final models, and running all-test inference:

```bash
.venv/bin/python -m uvicorn viralscope.demo_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default and is not exposed to the network. Dataset contains the full-train analyses, separately labeled seven-shard preview, and unlabeled test prediction distributions. Models contains the chronological, user-disjoint, and temporal comparisons and holdout diagnostic plots. Solutions records the work and decisions. Demo provides an editable synthetic example for requesting an estimate and explanation.

## Validation

The API was exercised in-process against all six final models; each returned HTTP 200, a finite prediction, and contribution entries. Metrics include the 61,123-row chronological holdout, the 66,983-row user-disjoint holdout with zero overlap, and three full-train cutoff-sensitivity runs. Batch inference covers all 180,581 public test rows; no test metrics are produced because labels are withheld. The UI shows prediction distributions and flags counts outside the verified train-label range without clipping or scoring them. Unit tests cover feature parity and ensure IDs are not returned. Run the full suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## License and Limitations

The page visibly displays **Powered by SMP Challenge** and the repository includes the required attribution in [Notice](../../Notice) and a copy of the Community License in [LICENSE-SMP-CHALLENGE](../../LICENSE-SMP-CHALLENGE.md). The all-train non-image cache, final models, and test prediction output are included in this research repository under that license. The incorporated AUP still requires review before reuse; images are not included or displayed.

Predictions are historical Flickr benchmark estimates and should not be interpreted as causal claims, user judgments, or platform-wide forecasts. The chronological holdout may share users across partitions; the separate group holdout is user-disjoint but not time-ordered. The public test predictions cannot be scored without labels. Images remain deferred pending license review and a separate model milestone.