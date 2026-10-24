# Measured MVP results

The complete supplied input was processed locally: **5,000 domains attempted**, **3,970 screenshots captured**, **1,030 failed visits**, and **3,564 nonblank screenshots embedded**. The 406 other captures failed the automated quality filter. CPU inference and browser execution used no paid compute or APIs.

1. **Popularity and visual homogeneity:** top-10k screenshots had mean within-tier cosine 0.362, versus 0.343 for the 100k-500k tier, an observed difference of 0.018. The ordering is not monotonic. Capture rates vary from 73.4% to 83.5%, so the comparison is descriptive and subject to selection bias.
2. **Lookalikes:** 409 unique retrieved pairs met the cosine >= 0.98 candidate threshold. A reviewed pair, 5movierulz.solutions and 5movierulz.ventures, has identical PNG hashes. Other high-similarity pairs share access-block/error templates; similarity does not establish common ownership.
3. **Outcome classifications need validation:** the seeded visual audit reviewed 80 captures; two were uncertain. Agreement on the other 78 was **62.8%**. Cookie predictions had precision 0/11 and for-sale predictions 0/6 in this scored sample. The raw predicted category counts cannot be read as true prevalence.
4. **Representation limits:** 0912sim.ir and adityanewsnetwork.com have cosine 0.915 because both show 403 pages, which says little about their intended sites. The logo-only 000h555.com splash is predicted as for_sale despite having no visible sale offer. Small text, center cropping, closed labels and capture artifacts limit semantic interpretation.

The visual audit was performed by **Codex assistant inspecting rendered screenshots**, not by an independent human. Labels, row-to-image bindings, notes, uncertain cases and reviewer attribution are preserved in `data/`. The audit was not used to tune the category prompts. Clustering is exploratory: k=12, cosine silhouette 0.100, indicating weak separation.

The four-page report is `reports/findings.pdf`. `outputs/submission_check.json` and `outputs/readiness.json` pass the mechanical integrity and completion gates. See the README for commands, the reproducibility archive and limitations. Binary screenshots and vectors are available locally and in the image-inclusive archive; they are excluded from Git.
