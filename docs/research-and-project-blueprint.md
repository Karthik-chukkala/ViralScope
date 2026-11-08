# ViralScope Research Findings and Project Blueprint

## 1. Purpose and Main Recommendation

ViralScope estimates the popularity of a social post before it is published and explains its prediction. Part 1 will be a reproducible text-and-context benchmark study with a working local inference website built around SMPD-Image training data. The image branch is deferred until the metadata pipeline, baselines, evaluation, and demo are complete; no image corpus download is needed for this milestone.

Use the benchmark's provided popularity label as the primary regression target unless its dataset card and files establish a different definition. Define a separate "viral" outcome only as a secondary task with a precise training-derived threshold. Do not equate predicting a historical Flickr score with forecasting current engagement on another platform.

The two papers are related, but the newer one is not a newer edition of the 2019 paper:

- [2019 paper](https://arxiv.org/abs/1910.01795): challenge and SMPD-Image overview, task definition, dataset construction, sequence/split setup, and official evaluation metrics.
- [2024 arXiv paper](https://arxiv.org/abs/2405.10497): a later retrospective analysis of challenge methods and performance trends. It was submitted to arXiv on May 17, 2024, and identifies the ACM Multimedia 2023 proceedings. It summarizes work from 2019 through 2023; it does not replace the 2019 task paper.

## 2. Research Question

### Primary question

Given information that would be available about a post and publisher before publishing at time $t$, how accurately can the system estimate the post's eventual benchmark popularity score?

### Secondary questions

1. How much predictive signal comes from text and prediction-time-safe structured/time context individually?
2. Does combining text and context improve held-out performance over the strongest single feature group?
3. Which inputs influence an individual prediction, and how consistent are explanations across examples?
4. Can a defined high-popularity subset be identified well enough to support a separate "viral candidate" demonstration?

The papers describe prediction for a particular future publication time. The Part 1 demo predicts from draft title/tags, category, and planned time; it does not use or display an image. An app that fetches a real post or claims live platform reach is a different product and outside the initial benchmark claim.

## 3. What the 2019 Paper Establishes

The 2019 paper introduces *Temporal Popularity Prediction*. For a new post $v$ from user $u$, the objective is to predict popularity $s$ if published at time $t$. Popularity is based on user attention; the challenge uses views as its basic indicator.

### Dataset and modalities

The paper describes SMPD as a Flickr-derived, multimodal collection of approximately 486K posts from around 69K users. It includes:

- **Visual:** photo images and human-curated hierarchical content categories.
- **Textual:** user-written title and custom tags.
- **Temporal:** post timestamps and ordered user-post activity.
- **Spatial:** optional location coordinates and a geo-accuracy level.
- **Publisher/context:** user and photo metadata.

The paper reports 11 first-level, 77 second-level, and 668 third-level categories, approximately 250K custom tags, and about 480 days of temporal coverage. These are historical paper statistics, not a guarantee that every later dataset release contains identical rows or fields.

### Temporal sequence hint

Posts from a user are represented in chronological order as a user-post sequence. The 2019 evaluation description says it uses a time-related partition, forms sequences with a window length of 10, and divides sequences 2:1 for training and testing. This is a key methodological hint: popularity prediction is not just an independent row-wise task. Preserve temporal ordering and verify the actual supplied split and user overlap before building a replacement split.

The paper reports approximately 305K training posts and 181K test posts. It describes the challenge test set as unpublished/withheld for evaluation. A public training split should not be casually called the official labeled test set.

### Popularity normalization

The 2019 paper's equation, verified from the MathML/TeX annotation in the [arXiv HTML](https://arxiv.org/html/1910.01795v2#S2.E1), is $s = \log_2(r/d) + 1$, where $r$ is a photo's view count and $d$ is the number of days since posting. As typeset, the `+1` is outside the logarithm; do not silently move it inside to obtain a different formula. The HF card describes `label` as a popularity score/log-views, but the release's exact calculation, snapshot time, rounding, and edge-case handling still need verification. Use the supplied label as the modeling target unless that audit establishes otherwise.

### Official evaluation

The 2019 paper evaluates:

- **Spearman rank correlation (SRC / Spearman's rho):** whether predicted scores order posts similarly to actual scores. Higher is better.
- **Mean absolute error (MAE):** typical absolute error in the label's units. Lower is better.

These are complementary: ranking can be good while score estimates are off, or vice versa. A user-facing app should define both in plain language and state which metric matches the challenge.

## 4. What the 2024 arXiv Paper Adds

The later paper reviews task/data context and summarizes challenge approaches from multiple years. Its main value to ViralScope is a map of model families, reported baselines, and the gradual shift from handcrafted/fused features toward pretrained multimodal representations.

### Approaches and participant hints

The overview discusses four broad approach families:

1. **Ensemble and boosted regressors:** SVM, logistic regression, decision trees, random forests, LightGBM, XGBoost, and CatBoost are cited as ways to combine engineered or extracted features. Later approaches often use separate text and visual models and combine their predictions.
2. **CNN visual representations:** CNNs such as ResNet-101 are used to extract image features, which are passed to a downstream regressor.
3. **Sequence models:** RNN-based approaches use chronological information and historical post behavior to model temporal context.
4. **Attention and transformers:** pretrained language and vision-language models provide contextual embeddings; later methods align title, tags, and image information or use contrastive learning.

The paper's references point to three notable 2019 directions:

- Junhong Chen et al., *Social media popularity prediction based on visual-textual features with XGBoost*: image/text feature fusion plus boosted regression.
- Keyan Ding et al., *Social media popularity prediction: A multiple feature fusion approach with deep neural networks*: deep feature fusion. The overview says a BERT model was used in a 2019 approach to obtain 768-dimensional text features, outperforming simpler one-hot/Word2Vec representations in that setting.
- Chih-Chung Hsu et al., *Popularity prediction of social media based on multi-modal feature mining*: multimodal feature mining.

The 2024 overview is a survey, not a complete reproduction guide for every 2019 participant. It does not provide enough details to recreate those systems exactly, nor should its broad descriptions be presented as a full list of 2019 teams or their private implementation settings. These works are sound ideas for controlled experiments, not guaranteed recipes or expected results.

### Reported baseline table

The later paper gives the following SMPD results:

| Features / approach | Regressor | SRC | MAE |
| --- | --- | ---: | ---: |
| ResNeXt-101 | LightGBM | 0.322 | 1.87 |
| BERT (base) | LightGBM | 0.538 | 1.65 |
| GPT-2 | LightGBM | 0.574 | 1.59 |
| FastText | CatBoost | 0.622 | 1.49 |
| FastText + ResNeXt | CatBoost | 0.700 | 1.40 |
| Winner 2020-A | CatBoost | 0.674 | 1.36 |
| Winner 2020-B | CatBoost | 0.704 | 1.37 |
| Winner 2022 | CatBoost | 0.710 | 1.24 |

These are the values as transcribed in the paper. They motivate testing modality-specific representations and fusion. They are **not directly comparable** to ViralScope unless dataset version, label, split, preprocessing, and evaluation protocol match. The paper also describes leaderboard progress from roughly 0.59 to 0.77 SRC over years; do not treat that range as a guaranteed target for this project.

## 5. Dataset Version and Access Checks

The [Hugging Face dataset page](https://huggingface.co/datasets/smpchallenge/SMP-Image) describes a training split of about 306K rows with image paths, labels, titles/tags, category, timestamps, location, and user/profile fields. The full pinned-train label scan found 305,613 valid labels, mean 6.4055 and population SD 2.4730, matching the 2019 paper's rounded train statistics. An initial label-only viewer audit found every label was `-1` in 381 test rows sampled across four offsets. Treat the public `test` split as unlabeled and do not compute test metrics from its placeholders unless the dataset documentation or official evaluator establishes otherwise. Details are in [Dataset Access and Audit](modules/dataset-access-and-audit.md) and [Cleaning and EDA](modules/cleaning-and-eda.md).

Published challenge-page snapshots report differing image totals (486K in one description and 680K in another). These may refer to different snapshots or scopes; do not combine them with historical paper counts. For Part 1, record the exact HF repository/config/revision, split names, row counts, files actually downloaded, and license.

Before modeling, verify:

- Review the SMP Challenge Community License and its incorporated AUP before use or publication. For any third-party distribution covered by the agreement, include the agreement and required attribution notice, and prominently display "Powered by SMP Challenge" on the related site/UI/documentation. Keep dataset images out of a public demo until the supervisor confirms the license conditions for that use. Do not use the dataset or its outputs to develop or enhance a foundational generative AI model.
- Whether image paths resolve in the downloaded release and what to do with missing or inaccessible images.
- `label` type, range, missingness, and documented meaning. Confirm whether it is already normalized.
- Duplicate rows, duplicate post IDs, duplicate images, and whether user IDs cross the supplied splits.
- The timestamp unit/time zone and which features existed before the target post was published.
- Whether user/profile fields are time-stamped snapshots or may contain information added after the post.

Hugging Face browser authorization is already complete, as confirmed by the project owner. Browser authorization may not automatically authenticate the Python client inside WSL, so first verify access by loading dataset metadata and a small sample. If the client requires separate authentication, use a secure local Hugging Face CLI login; do not request, copy, or store the token in project files, notebooks, logs, or screenshots.

## 6. Leakage-Safe Experimental Design

### Prediction-time feature rule

For every feature, ask: *Could this value have been known at the proposed publication time?* Exclude future engagement and post-publication state. Be cautious with profile photo counts, user descriptions, user histories, popularity aggregates, and metadata collected after the event. If historical user features are included, reconstruct them using only earlier posts in timestamp order.

### Split protocol

1. Prefer the dataset's documented official training split and reproduce its row counts.
2. Create a validation subset from training for feature/model selection; preserve chronology and relevant user-sequence structure.
3. The current HF `test` split has sampled `-1` labels; treat it as unlabeled and do not calculate SRC/MAE from those values. Confirm the sentinel convention with release documentation.
4. Evaluate on a documented chronological or user-aware local holdout from labeled training data. Call it local validation/holdout, not the official test set. Only report official test metrics if genuine labels are returned by an authorized evaluator.
5. Report whether users, images, and time periods overlap across partitions. Explain the intended generalization setting.

Never select the model or tune parameters on the final test set. Training metrics are useful for diagnosing under/overfitting but must not be presented as generalization performance.

### Suggested model ladder

Start small and keep each experiment identifiable:

1. **Sanity baselines:** median predictor; simple category/time structured regressor.
2. **Text-only:** title and tags with TF-IDF plus Ridge.
3. **Structured-only:** category and carefully derived planned-time features with CatBoost.
4. **Text + context:** combine the text representation and eligible structured features, then compare against the single-group baselines.
5. **Deferred image branch:** if time and license review permit after Part 1, extract frozen pretrained visual features and cache embeddings; only then test image-only and multimodal fusion.
6. **Optional temporal extension:** sequence features from each user's earlier posts, strictly using data before the target timestamp.

The 2019/2024 papers support these as research-grounded comparisons. Large-model fine-tuning, video, graph neural networks, and elaborate contrastive learning are optional future work, not prerequisites for a credible Part 1.

### Metrics and analysis

Primary challenge metrics: SRC and MAE. Add RMSE (penalizes larger errors) and $R^2$ (variance explained, interpreted with care) as complementary regression metrics. Report these on the local labeled holdout, alongside sample counts and label distributions. Report final/official test metrics only if genuine test labels are available; the current HF test sample contains `-1` placeholders and is not scoreable. Include:

- Per-category and popularity-range performance.
- Error distribution/residual plots and prediction-versus-target plots.
- Ablations: text only, image only, structured only, and fused; optionally remove each feature group from the chosen model.
- A basic baseline comparison and train-versus-validation gap.
- Runtime/model size if they matter to the demo.

Avoid calling any regression score "accuracy." If a secondary viral classification is added, define the threshold from training labels only (for example, top 10%), then report precision, recall, F1, and PR-AUC. Do not use MAPE as a default for a target that may be zero or near zero.

## 7. Explainability Plan

Explainability should describe the behavior of the trained model, not claim causal facts about people or content.

- For CatBoost/LightGBM, use SHAP or native feature contributions for global and per-example explanations.
- For TF-IDF, show influential words/tags with their direction and clarify that correlated terms can share or distort importance.
- For image inputs, use a suitable saliency/occlusion method only if it can be implemented and validated; present it as a model sensitivity view, not proof of semantic reasoning.
- Show at least one counterfactual-style sensitivity check, such as comparing the same sample with/without tags or a modality. Do not imply the modified post's real-world outcome is known.
- State that explanations are associations within historical Flickr data and that correlation does not establish causation.

For the review demo, display the predicted historical benchmark score, selected model/version, the top contributing available feature groups, and a concise uncertainty/limitations note. Only display confidence intervals if they are actually estimated and evaluated; otherwise, do not label a score as confidence.

## 8. Website and Working Demo

The website should be a project instrument and review presentation, not just static marketing pages. Recommended sections:

1. **Project overview:** problem, prediction moment, target, intended use, and what the system does not claim.
2. **Dataset:** provenance, version, license, split/count summary, fields, missingness, image access, and known limitations.
3. **Exploratory analysis:** target distribution, modality coverage, category imbalance, temporal trends, text/tag summaries, image quality/availability, and privacy-conscious location analysis.
4. **Pipeline:** data validation, cleaning choices, feature groups, split protocol, and saved experiment configuration.
5. **Models and evaluation:** model comparison table, understandable metric definitions, validation/test results, ablations, and error slices.
6. **Explainability:** global feature contributions plus a few carefully selected example predictions.
7. **Interactive prediction:** title/tags, category and planned post time; validate inputs and show prediction/explanation using the saved model. The first demo does not request an image because the Part 1 model does not use one.
8. **Limitations and reproducibility:** data age/platform, split caveats, ethical/license constraints, hardware, dependencies, and commands/config needed to reproduce results.

For Part 1, "live" means inference runs in the local app, not live platform data or actual future engagement. Training/EDA/evaluation outputs are precomputed and served by the implemented local FastAPI app with a responsive HTML interface. The demo remains bound to loopback by default and is not ready for public deployment.

## 9. Part 1 / Part 2 Completion Definition

Treat "60%" as a list of working artifacts, not an arbitrary progress claim. A defensible Part 1 milestone includes:

- Exact dataset release and license recorded; reproducible download/load path.
- Data validation and cleaning documented with counts before/after each meaningful step.
- Broad EDA with saved charts and interpretations, not charts alone.
- Leakage-aware train/validation protocol and at least three meaningful text/context baselines or model variants.
- Feature-group ablation and a selected model with genuine held-out metrics.
- Working local UI for the project narrative and single-sample inference.
- An explanation view, model limitations, and reproducibility instructions.
- A short review script and example inputs that work offline after model/data artifacts are cached.

### Part 2: sensible follow-through

- Improve temporal/user-history modeling or multimodal fusion if Part 1 evidence supports it.
- Test robustness across time, category, and user groups; assess uncertainty and calibration if the interface needs it.
- Add a separately licensed/collected dataset or platform only if access, consent, and scope permit.
- Deployment, API hardening, monitoring, and broader evaluation.
- Consider video as an independent extension using the separate SMPD-Video track, not as an assumed property of SMPD-Image.

## 10. Hardware and Environment Readiness

The [current setup record](../files/setup.txt) shows WSL Ubuntu 24.04, Python 3.12, an activated `.venv`, and installed analysis/model/API packages: pandas, NumPy, scikit-learn, SciPy, PyArrow, Hugging Face `datasets`, CatBoost, Matplotlib, Seaborn, SHAP, FastAPI, Uvicorn, PyTorch, and torchvision. The transcript reports FastAPI 0.162.1 and PyTorch 2.15.0+cu132. Its verification command reports `torch.cuda.is_available() == True` and detects an NVIDIA GeForce RTX 4050 Laptop GPU. These are recorded setup results; the project should still run a smoke test in the active environment before relying on them during implementation.

That is a strong starting environment for SMPD-Image. With 16 GB system RAM and 6 GB VRAM, favor streaming/columnar loading, batch image encoding, cached embeddings, and moderate batch sizes. The GPU is useful for pretrained feature extraction and small experiments; it is not a reason to train a large vision-language model from scratch. Keep the API/UI responsive by loading the final model once and avoid loading the full image corpus into RAM.

The setup log records very new package versions. CatBoost, scikit-learn, PyArrow, FastAPI, and Uvicorn were verified in the active environment; image inference was not needed for this text/context milestone. Keep a pinned dependency file after the first successful reproducible environment rather than relying only on an installation transcript.

## 11. Module Documentation Standard

Every implementation module and meaningful subpart will have a corresponding detailed document under `docs/modules/`. The project-level blueprint remains the overview and links to the module documents. Keep the docs synchronized with actual implementation; mark planned behavior separately from verified behavior.

Each module document should cover:

- **Purpose and boundaries:** what the module does, why it is needed, and what it does not do.
- **Research basis:** relevant paper insight, terminology, and design rationale.
- **Inputs and outputs:** file/API formats, schemas, required and optional fields, and where artifacts are written.
- **Processing and decisions:** step-by-step flow, cleaning/feature rules, assumptions, and prediction-time leakage checks.
- **How to run:** exact command, environment/configuration, expected runtime or resource needs, and expected output.
- **Validation:** tests or checks, dataset counts, metrics, and how to interpret the results. Separate planned checks from completed checks.
- **Limitations and troubleshooting:** known edge cases, failure modes, privacy/license considerations, and safe recovery steps.

Suggested module documents as the project grows: dataset access and audit; cleaning and EDA; feature engineering; model families and experiments; evaluation; explainability; API; UI/demo; and reproducibility/deployment. Current module docs: [Dataset Access and Audit](modules/dataset-access-and-audit.md), [Cleaning and EDA](modules/cleaning-and-eda.md), [Modeling Baselines](modules/modeling-baselines.md), and [Local Demo](modules/local-demo.md). Add each next module document as its implementation begins.

## 12. Current Implementation Decisions

1. Use pinned `smpchallenge/SMP-Image` revision `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`; retain the supplied `label` as target. Exact label snapshot/formula semantics are not fully verified.
2. Do not score the public test split: sampled labels are `-1`. Report both a chronological holdout and a separate random user-group holdout from the same seven-shard labeled sample. The user-group split has zero user overlap but is not time-ordered; a combined future-time/unseen-user protocol remains future work.
3. Part 1 uses title, tags, category/subcategory/concept, and planned publication time. Profile snapshots, engagement counts, IDs, locations, and images are excluded.
4. The review demo is one local FastAPI app serving its UI and inference API. It does not expose a public network interface by default.
5. Continuous regression is primary; no separate viral threshold/classifier is implemented.
6. Public deployment, model/data redistribution, and image use remain gated on supervisor review of the complete license and AUP.

These decisions keep the current results reproducible and bound their claims while the larger data and review milestones remain open.

## 13. References

1. Bo Wu, Wen-Huang Cheng, Peiye Liu, Bei Liu, Zhaoyang Zeng, and Jiebo Luo. 2019. *SMP Challenge: An Overview of Social Media Prediction Challenge 2019*. ACM Multimedia 2019. DOI: 10.1145/3343031.3356084. [arXiv:1910.01795](https://arxiv.org/abs/1910.01795).
2. Bo Wu, Peiye Liu, Wen-Huang Cheng, Bei Liu, Zhaoyang Zeng, Jia Wang, Qiushi Huang, and Jiebo Luo. *SMP Challenge: An Overview and Analysis of Social Media Prediction Challenge*. ACM Multimedia 2023; arXiv:2405.10497, submitted May 17, 2024. DOI: 10.1145/3581783.3613853. [arXiv:2405.10497](https://arxiv.org/abs/2405.10497).
3. Junhong Chen, Dayong Liang, Zhanmo Zhu, Xiaojing Zhou, Zihan Ye, and Xiuyun Mo. 2019. *Social media popularity prediction based on visual-textual features with XGBoost*. ACM Multimedia 2019, 2692-2696. Cited as reference [3] in the later overview.
4. Keyan Ding, Ronggang Wang, and Shiqi Wang. 2019. *Social media popularity prediction: A multiple feature fusion approach with deep neural networks*. ACM Multimedia 2019, 2682-2686. Cited as reference [9] in the later overview.
5. Chih-Chung Hsu et al. 2019. *Popularity prediction of social media based on multi-modal feature mining*. ACM Multimedia 2019, 2687-2691. Cited as reference [13] in the later overview.