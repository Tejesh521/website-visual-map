# Visual Map of 5,000 Websites

A local Python 3.11 command-line research pipeline. It visits each supplied homepage, records capture outcomes (including failures), embeds nonblank viewport screenshots with pretrained OpenCLIP, and produces similarity, cluster, popularity-tier and zero-shot analyses plus a four-page PDF. No paid API, cloud compute, dashboard or model training is used.

## Install

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m playwright install chromium
python -m pytest -q
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` also pins the transitive environment used for this run. CPU PyTorch wheels vary by platform. First use downloads free model weights (~600 MB) from Hugging Face; allow several GB for dependencies, browser, weights and screenshots. This project was executed on macOS ARM64. Linux may need `python -m playwright install --with-deps chromium`.

## Run

The supplied `data/domains.csv` has exactly 5,000 domains in five equally sized popularity tiers. The original root `domains.csv` is retained unchanged. Default input is always `data/domains.csv`.

```sh
python run_pipeline.py --concurrency 12 --timeout 12
```

That is the protocol used in this workspace (one-second settle, one retry, 1280 x 800 viewport). The generic default is concurrency 6 and timeout 20 seconds. Omitted timeout, settle and retry options reuse the saved protocol, so `python run_pipeline.py` also safely resumes this collection. A protocol mismatch fails explicitly instead of mixing methods. Concurrency may change without changing the protocol. Long-running visits execute locally and depend on your network; do not assume a fixed completion time.

For a fresh small pilot, add `--limit 30 --to-stage quality`, then run the same command without `--limit`. `--limit` means at most that many *pending* domains, not the first N rows. Work is interleaved by tier. The capture journal is flushed and synced after each completed domain. Re-running capture skips successful images whose hashes still match, and skips recorded failures unless `--retry-failed` is explicitly passed to `src/capture.py`. Missing or changed successful screenshots are recaptured. Completed outputs remain usable after interruption; a crash can repeat the in-flight visits.

Stages can also run independently, in this order:

```sh
python src/validate.py
python src/capture.py --concurrency 12 --timeout 12
python src/quality.py
python src/embed.py
python src/similarity.py
python src/cluster.py
python src/tier_analysis.py
python src/zero_shot.py
python src/visualize.py
python src/report.py
```

`python run_pipeline.py --from-stage quality` recomputes analysis from the current captures. Embedding reuse requires an identical quality manifest, model identifier and preprocessing protocol. Do not run analysis stages while a collection is changing if you need a final, consistent report. To start a different capture protocol, archive the current metadata, screenshots, embeddings and outputs first; do not mix collections silently.

## Review evidence and finalize

Automatic nonblank checks do **not** certify an intended homepage: CAPTCHAs, HTTP errors, login screens and cookie dialogs are valid *capture outcomes*. HTTP status is retained independently of screenshot success. HTTPS certificate errors are not bypassed. Navigation is bounded, with one HTTP fallback after a failed original origin; all attempted URLs, timing and errors are in the journal. No scrolling, clicking, consent dismissal, login or CAPTCHA solving occurs.

1. Inspect `outputs/contact_sheet.png`, `neighbor_pairs.png`, `outliers.png` and `category_samples.png`.
2. `zero_shot.py` selects a seeded random audit in `outputs/audit_sample.csv` and adds its rows to `data/manual_labels.csv`. As the collection grows, the selected sample changes; old review rows are preserved, and metrics use only the current sample. Labels reset if an image hash changes. Inspect each original screenshot, fill `manual_label`, `notes`, and `reviewer`. Allowed labels: `content`, `login`, `captcha`, `cookie`, `for_sale`, `error`; use `uncertain` for unresolved images. These labels describe the dominant visible capture outcome. The sample is independent of predictions; do not substitute easier examples.
3. Record observed lookalikes and a concrete embedding failure in `data/research_notes.json`, with `text` and matching `site_ids` under `lookalikes` and `embedding_failure`. A reviewer should be able to open those exact screenshots to verify the observation. If a model assistant performs the visual audit, disclose that reviewer and do not call it independent human ground truth.
4. Rerun `python run_pipeline.py --from-stage zero_shot`. Audit metrics and the report update. Existing manual labels are preserved when their image hash still matches.

The report explicitly remains **DRAFT** until all 5,000 input rows have recorded attempts, at least 180 usable images have embeddings, the 80-image random audit has reviewer-attributed labels (uncertain reviews count as reviewed but are excluded from agreement), and qualitative evidence references real images. `outputs/readiness.json` records these gates. Passing a mechanical gate does not replace critical review of the claims.

## Outputs and methods

| Artifact | Meaning |
|---|---|
| `metadata/validation_summary.json` | Source and normalized CSV hashes, row/tier counts |
| `metadata/capture_attempts.jsonl` | Durable append-only collection journal, including every retry history |
| `metadata/capture_manifest.csv` | One row per input domain; latest result; blank status means not yet attempted |
| `metadata/quality.csv` | Hash, grayscale entropy/std, nonblank decision for each input domain |
| `embeddings/embeddings.npy` | Float32, L2-normalized image vectors in manifest row order |
| `embeddings/embedding_manifest.csv` | Exact embedding-row to screenshot mapping and image hashes |
| `embeddings/model.json` | Model/preprocessing identifier, input signature, count, dimensions, versions |
| `outputs/nearest_neighbors.csv` | Up to five cosine neighbors per image, excluding self even with ties |
| `outputs/near_duplicates.csv` | Unique retrieved neighbor pairs above 0.98; candidates, not proven duplicates |
| `outputs/clusters.csv` | KMeans labels and representatives nearest to normalized centroids |
| `outputs/projection.csv` | Seeded UMAP coordinates; PCA only for fewer than ten images |
| `outputs/tier_analysis.csv` | Coverage, exact within-tier mean cosine, 300 site-bootstrap 95% intervals |
| `outputs/cross_tier.csv` | Exact average cosine between different tiers |
| `outputs/zero_shot_labels.csv` | Dominant predicted page type, all prompt cosine scores, top-two margin |
| `outputs/classification_audit.json` | Review coverage, agreement, per-class precision and recall |
| `reports/findings.pdf` | Four pages, with explicit draft/readiness status |

OpenCLIP `ViT-B-32`, weights `laion2b_s34b_b79k`, runs on CPU with batch size 8 and four Torch threads. RGB images use the model's default resize/center crop and normalization. Embeddings are unit-normalized before cosine analysis. Center cropping can miss page edges; CLIP is not an OCR or website-semantic oracle. The checkpoint is pinned to a Hugging Face commit and SHA-256 in `config.py`; loading verifies its bytes. The revision and hash are archived in `metadata/model_weights.json`. Weights are cached by Hugging Face outside the repository.

KMeans uses seed 42, ten initializations, and k in {2, 3, 5, 8, 12}, excluding invalid k. The highest cosine silhouette wins; the score is exploratory and the geometry is KMeans on unit vectors. Silhouette uses at most 1,500 seeded sampled points. UMAP uses cosine, 15 neighbors (or n-1 for small samples), min_dist 0.1, one job and seed 42. Do not interpret projected distances as original cosine values.

Tier similarity means use every distinct within-tier pair analytically without materializing an n-by-n matrix. Confidence intervals resample *websites*, not correlated pairs. Coverage denominators always include every input row. The random pair sample is used only for a descriptive standard deviation. Intervals and associations cannot establish a causal popularity effect and do not correct for capture-selection bias.

Zero-shot category scores average normalized embeddings of two English prompts per category, then choose the highest cosine. Scores are not calibrated probabilities. Competing categories can overlap (for example, a login page with a cookie dialog). Report predicted prevalence among embedded images, with manual validation and uncertainty, rather than claiming true prevalence across all 5,000 websites.

## Reproducibility and limitations

Code, fixed seeds, prompts, metadata and the locked environment reproduce analysis over archived inputs. Live recapture is inherently time-, region-, network- and browser-dependent; it cannot promise byte-identical pages. Screenshot and embedding binaries are ignored by Git because of size. Preserve them with `python src/package.py` for a complete analysis handoff, together with the hash inventory. Never describe a repository containing only code as a frozen-data reproduction package.

`python src/package.py --include-images` creates `reports/reproduction_bundle.tar.gz` including screenshots and embeddings. It also creates `metadata/artifact_hashes.json`; validate with `python src/package.py --verify`. The archive includes source, input, reviewed labels, outputs, environment versions and protocol. It excludes the virtual environment, model cache, Git internals, logs, and itself. Collection concurrency was increased from 12 to 24 to 48 after checking local memory; final timing and retry history remain in the journal. No paid compute is used; local electricity/storage/network costs are not estimated.

Tests cover CSV validation (duplicates, host mismatches, IDNs, origin ports), blank/corrupt image filtering, a real local Chromium visit and resume, neighbor self-exclusion with tied vectors, and stale embedding rejection. Run `python -m pytest -q`. External network successes are intentionally not asserted by tests.

Primary references: [Playwright screenshots](https://playwright.dev/python/docs/screenshots), [Playwright Page API](https://playwright.dev/python/docs/api/class-page), [OpenCLIP](https://github.com/mlfoundations/open_clip), [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html). Screenshots are research evidence from their respective sites; no ownership or redistribution rights over website content are asserted.

Final integrity check: `python src/check.py --strict`. This verifies input-to-capture mapping, usable-image-to-vector mapping, embedding row order, every embedded screenshot hash, analysis provenance, four PDF pages, and readiness gates. See `MILESTONES.md` for the completion evidence associated with each roadmap milestone.
