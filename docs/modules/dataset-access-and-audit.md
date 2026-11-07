# Module 1: Dataset Access and Audit

## Status

**Metadata-only CLI, full label-column audit, and bounded image check: verified.** Seven evenly spaced train shards have also been range-read for metadata and cached locally with selected non-image columns for baseline training. The full Parquet dataset and image corpus have not been downloaded.

## Purpose

Identify the exact SMPD-Image release ViralScope will use, establish authorized and reproducible access, and verify its schema, split sizes, license, labels, and image availability before cleaning or modeling.

This module owns dataset provenance and initial quality evidence. It does not own feature engineering, model training, EDA conclusions, or UI implementation; those will be documented as their modules are started.

## Research Basis

The 2019 paper describes SMPD as a Flickr-derived benchmark with about 486K posts and a temporal train/test protocol. The later challenge-analysis paper summarizes the benchmark and participant approaches. The HF repository inspected here is `smpchallenge/SMP-Image`; its release must be treated as a specific artifact and verified independently rather than presumed identical to every paper or website snapshot.

Sources: [2019 paper](https://arxiv.org/abs/1910.01795), [2024 arXiv paper](https://arxiv.org/abs/2405.10497), [HF dataset page](https://huggingface.co/datasets/smpchallenge/SMP-Image).

## Environment and Access

The project uses the existing WSL `.venv` and `datasets` / `huggingface_hub` packages recorded in [setup](../../files/setup.txt). The project owner reports completing Hugging Face browser authorization. A metadata probe inside WSL found a locally configured HF CLI credential without printing its value, and successfully read repository metadata and the dataset card.

Browser login and a WSL Python credential are not necessarily the same session. For repeatability, run commands from the project root using `.venv/bin/python`. The current sample cache uses pinned remote Parquet range reads; it does not fetch whole image-bearing shards. Never put a token in a command, source file, notebook, screenshot, or documentation. If access later fails, check the local HF login without exposing credential contents.

## Verified Metadata

Repository: `smpchallenge/SMP-Image`.

Revision observed during this audit: `f131dba653a05c0e3c6294388f81ed3fbd9c71a0`. Recheck this SHA before downloading or training, since the repository's main branch can change.

The authenticated HF dataset-server size endpoint, cross-checked against the pinned repository file metadata, reports:

| Split | Rows |
| --- | ---: |
| train | 305,613 |
| test | 180,581 |

The repository metadata reports 80 files and these feature names:

`image`, `Uid`, `Pid`, `img_filepath`, `label`, `Postdate`, `Longitude`, `Latitude`, `Geoaccuracy`, `Category`, `Subcategory`, `Concept`, `Alltags`, `Title`, `Mediatype`, `Mediastatus`, `Pathalias`, `Ispublic`, `photo_count`, `ispro`, `timezone_offset`, `photo_firstdate`, `photo_firstdatetaken`, `timezone_id`, `user_description`, `location_description`.

The generated report confirms `label` is `float32` and `image` is an HF image feature. The size endpoint reports `partial: false` and 486,194 total rows. This is distinct from the `/statistics` endpoint, which reports `partial: true` and covers only 64,551 train rows in its label summary.

Repository metadata lists 78 Parquet shards. Their reported sizes total approximately **37.78 GiB**: 25,658,940,071 bytes for train and 14,911,095,984 bytes for test. The pinned builder exposes feature metadata but omits split counts; the audit CLI gets counts from the authenticated viewer size endpoint and refuses to write a report if its split byte totals do not match the pinned repository metadata. The viewer endpoint does not expose a revision, so the CLI first verifies that the repository's current main SHA matches the requested revision. Do not fetch every shard just to inspect a sample.

### License terms found in the HF card

The card metadata says `license: other`, `license_name: smp-license`, and links to `LICENSE.md`. No separate `LICENSE.md` or AUP file appeared in the repository file listing; the README embeds the **SMP Challenge Community License Agreement**, last updated March 17, 2025, in its gated prompt.

The agreement states that it grants research and non-commercial use, including use, reproduction, distribution, and creation/modification of derivative works, subject to its terms. It also describes a commercial-use license, registration requirements, and a threshold above USD 50,000 annual revenue at which a separate license must be requested. For a third-party distribution of the materials/derivatives, or a product/service using them, it requires providing the agreement, retaining this attribution notice in a `Notice` file, and prominently displaying **"Powered by SMP Challenge"** on the related site/UI/documentation:

> This SMP Challenge dataset is licensed under the SMP Challenge Community License, Copyright © SMP Challenge. All Rights Reserved.

It prohibits use of the materials, derivatives, or their outputs to develop or enhance a foundational generative AI model. The agreement incorporates an Acceptable Use Policy (AUP), but that policy was not located in the repository files inspected.

These are notes from the card, not legal advice. Before a public review website or distribution, review the complete current agreement and AUP with the project supervisor and confirm whether screenshots, sample images, model artifacts, or a hosted demo count as distribution under the terms. Do not assume that browser access alone grants permission to republish image content.

These split sizes are close to the paper's approximate 305K/181K counts. This numerical agreement does not prove that the split protocol, contents, labels, or image files are identical to the official challenge release.

### Bounded label-only viewer check

The HF dataset viewer was queried for only the `label` column, requesting 100 rows per offset; no titles, user fields, or image bytes were printed or saved. The samples were taken at offsets 0, 1,000, 150,000, and 305,500 for train, and 0, 1,000, 90,000, and 180,500 for test.

| Split | Sampled rows | Label observation |
| --- | ---: | --- |
| train | 400 | All sampled labels were non-`-1` numeric values; values differed across offsets. |
| test | 381 | Every sampled label was exactly `-1.0`, including the final partial batch. |

This is strong evidence that the public `test` split uses `-1` as a withheld-label placeholder. Treat it as **unlabeled** unless the dataset documentation or official evaluator confirms otherwise. Do not compute or publish test MAE/SRC using these placeholders. Create a chronological/user-aware local validation holdout from the labeled training split, report its metrics as local validation, and reserve the public test split only for inference/submission if the challenge provides an evaluator. The samples do not prove every row has the same value; confirm the sentinel convention from dataset documentation before final reporting.

The HF statistics endpoint returned `partial: true` and summarizes only 64,551 training rows. Its sampled mean (7.14298) differs from the full training mean; do not present this endpoint as full-training EDA. The full pinned-train scan in [Cleaning and EDA](cleaning-and-eda.md) reports mean 6.4055 and population SD 2.4730, matching the paper's train mean 6.41 and SD 2.47 after rounding.

The 2019 paper's normalization equation was verified in the equation's [arXiv HTML MathML/TeX annotation](https://arxiv.org/html/1910.01795v2#S2.E1): $s = \log_2(r/d) + 1$, with $r$ the view count and $d$ the days since posting. The `+1` is outside the logarithm in the published expression. The dataset card calls its `label` a popularity score/log-views, but the bounded checks do not prove that this dataset release applied that exact formula, which view snapshot/date it used, or how it handled boundary cases. Keep the provided `label` as the target and do not recalculate it until those details are established.

### Bounded image-field and decode check

The viewer returned a non-null `image` object and non-empty `img_filepath` for three sampled rows in each split. The image objects exposed `src`, `height`, and `width`. One training image was then retrieved from its viewer-provided source and decoded in memory with Pillow; it was not written to disk or displayed.

| Check | Result |
| --- | --- |
| Image retrieval | Successful for the single sampled train image |
| Content type / decoded format | `binary/octet-stream` / JPEG |
| Bytes read | 30,291 |
| Decoded dimensions | 333 x 500 |

This verifies one sample path, not corpus-wide image integrity, licensing for public display, or that all `img_filepath` values resolve in the Parquet release.

### Storage check

At audit time, the workspace filesystem reported 932.74 GiB free. The repository metadata reports 25,658,940,071 bytes of train Parquet and 14,911,095,984 bytes of test Parquet, about 37.78 GiB total. This suggests sufficient local capacity for a later planned download, but does not justify downloading the unlabeled test split. Recheck free space before fetching the train split and account for HF cache/working-copy overhead.

## Project Implementation

The metadata-only audit is implemented in [dataset_audit.py](../../viralscope/dataset_audit.py). From the repository root, run:

```bash
.venv/bin/python -m viralscope.dataset_audit
```

The command writes `artifacts/dataset-audit.json` with the pinned repository revision, split sizes, feature names/types, license identifier/link, and Parquet file sizes. It downloads metadata only, not dataset rows or image bytes. It requires the already-configured local Hugging Face credential for the gated size endpoint; the token is never printed or written to the report. The generated report is limited to repository metadata and does not contain post text, user fields, or image content.

Focused unit tests for the report helpers:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The latest run produced the row counts above and confirmed that the viewer's per-split Parquet byte totals match repository file metadata. The report records that its `test_labels_verified` scope is false; the separate bounded label sample below is not a full split audit.

## Remaining Checks

Carry these into the next data module or resolve before publication/model evaluation:

1. **License and terms:** the card's embedded agreement has been read; locate/review the incorporated AUP and confirm public-demo/display/distribution obligations with the project supervisor. The card links to `LICENSE.md`, but no separate file was listed.
2. **Artifact identity:** the observed revision and metadata report are recorded above. Use that exact revision for data processing; regenerate the report if changing the pin.
3. **Target semantics:** the full pinned-train scan confirms numeric `float32` labels, range 1.0-16.56, and mean/SD matching the paper's rounded training statistics. The paper's equation is $s = \log_2(r/d) + 1$, but matching aggregate statistics does not prove the exact view snapshot, rounding, or edge handling; do not overwrite supplied labels based on the equation alone.
4. **Split meaning:** sampled test labels are all `-1`; confirm the sentinel convention and split purpose from release documentation. Do not call it the official withheld challenge test set unless the release documentation establishes that.
5. **Rows and IDs:** in Cleaning & EDA, check null counts, duplicate rows/IDs, user overlap, image-path integrity, and split cardinalities on loaded data.
6. **Images:** one image decode passed; in Cleaning & EDA, measure missing/corrupt image rates before choosing an image-model subset or fallback policy.
7. **Prediction-time safety:** identify fields that could be post-publication snapshots (including user/profile metadata) and exclude or reconstruct them from prior-only history before modeling.
8. **Resource estimate:** estimate parquet/image storage and RAM needs before downloading the full image corpus; use streaming or bounded batches for initial inspection.

## Safety and Privacy

- Keep Hugging Face credentials outside the repository and avoid displaying them in terminal captures.
- Treat user IDs and profile descriptions as potentially identifying data; do not surface raw profiles in the website.
- Aggregate or omit exact coordinates in visualizations.
- Respect the dataset's image license and the platform's content/privacy expectations.
- Do not use post-publication values as predictors for a pre-publication forecast.

## Completion Criteria

This access module is complete when the license summary and revision are recorded; the metadata-only audit command is reproducible; split counts and storage totals are cross-checked; and bounded label/image access has been tested. Full missingness, duplicate, invalid-record, and prediction-time feature audits belong to Cleaning & EDA. Public-demo license/AUP review remains a publication gate. The complete dataset does not need to be copied into version control.

## Next Step

The access audit, full label scan, seven-shard metadata sample, initial text/context model ladder, and local review demo are implemented. The next work is to broaden EDA, improve evaluation with a user-aware check and additional temporal cutoffs, and review the complete license/AUP with the supervisor before any public deployment or image use. Do not score the public test split; its sampled `-1` labels are withheld placeholders.