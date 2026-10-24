# Milestone evidence

This checklist describes the required completion evidence. The final measured run status is in `outputs/submission_check.json` and `outputs/readiness.json`; source files alone do not prove collection or research completion.

| Milestone | Implementation / completion evidence |
|---|---|
| 1. Initialize | Python 3.11 environment, pinned direct and full dependencies, config, README, Git; `metadata/environment.json`; smoke tests |
| 2. Validate | `src/validate.py`; 5,000 unique domains and five equal tiers; input hashes; IDNs and origin ports preserved |
| 3. Capture | `src/capture.py`; fixed viewport, status, response timing, UTC capture time, redirect URL, screenshot hash |
| 4. Robust collection | bounded navigation, fresh contexts, HTTP fallback, append-only fsynced journal, resume, process lock; 5,000 completed domain records required |
| 5. Quality | decode, dimensions, entropy, grayscale variance and integrity checks; contact sheets and original-image review |
| 6. Embeddings | frozen pretrained OpenCLIP, CPU, RGB preprocessing, L2 normalization, exact row mapping and image hashes; at least 180 usable images required |
| 7. Similarity | cosine neighbors, near-duplicate candidates, outliers, seeded KMeans selection and UMAP; representatives |
| 8. Popularity | input/capture/embedding coverage, exact within-tier and between-tier means, site-bootstrap intervals, category sensitivity table |
| 9. Zero-shot | archived two-prompt categories, raw cosine scores and margins; random 80-image visual audit and disagreement metrics |
| 10. Findings | four measured report questions; concrete reviewed lookalike and failure evidence with screenshot IDs; capture bias disclosed |
| 11. Reproduction | README, dependency lock, seeds, input/config/model/environment hashes, binary-inclusive archive and verification command |
| 12. Submission | four rendered/inspected report pages; `python src/check.py --strict`; `python src/package.py --verify` |

Collection and analysis can complete automatically. The visual audit is attributed to its actual reviewer; a model-assisted review must not be represented as independent human ground truth. A four-page draft is available before all gates pass, and is labeled accordingly.

## Completed run

All twelve implementation milestones have supporting artifacts. The final run attempted 5,000 domains and embedded 3,564 nonblank captures; the 80-image assistant visual audit retained two uncertain outcomes. The four-page PDF was rendered and visually inspected. Strict integrity checks pass. This is an assistant-reviewed research submission; independent human validation remains a distinct future check, not a result claimed here.
