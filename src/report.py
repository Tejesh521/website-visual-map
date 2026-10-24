"""Exactly four pages. Draft until collection, embedding and manual-evidence gates pass."""

from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from common import *


def report():
    m, x = load_embeddings()
    cap = read("metadata/capture_manifest.csv")
    tiers = read("outputs/tier_analysis.csv")
    z = read("outputs/zero_shot_labels.csv")
    verify_analysis(
        [
            "outputs/tier_analysis.csv",
            "outputs/nearest_neighbors.csv",
            "outputs/zero_shot_labels.csv",
            "outputs/projection.csv",
        ]
    )
    quality_protocol = json.loads(require("metadata/quality_protocol.json").read_text())
    if quality_protocol["capture_manifest_sha256"] != digest(
        require("metadata/capture_manifest.csv")
    ):
        raise RuntimeError(
            "Captures changed after quality review. Rerun from quality after collection finishes."
        )
    figure_sources = json.loads(require("outputs/figure_sources.json").read_text())
    if any(digest(require(name)) != value for name, value in figure_sources.items()):
        raise RuntimeError("Figure inputs changed. Rerun visualize before report.")
    audit = json.loads(require("outputs/classification_audit.json").read_text())
    if audit["manual_labels_sha256"] != digest(
        require("data/manual_labels.csv")
    ) or audit["predictions_sha256"] != digest(require("outputs/zero_shot_labels.csv")):
        raise RuntimeError(
            "Manual labels or predictions changed. Rerun zero_shot, visualize and report."
        )
    summary = json.loads(require("outputs/cluster_summary.json").read_text())
    evidence_path = path("data/research_notes.json")
    notes = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    attempted = int(cap.status.isin(["success", "failed"]).sum())
    captured = int((cap.status == "success").sum())
    manual = read("data/manual_labels.csv")
    review_complete = audit["reviewed"] >= min(80, len(m)) and audit[
        "sample_size"
    ] == min(80, len(m))
    evidence_ok = all(
        notes.get(k, {}).get("text")
        and notes.get(k, {}).get("site_ids")
        and set(notes[k]["site_ids"]).issubset(set(m.site_id))
        for k in ["lookalikes", "embedding_failure"]
    )
    ready = (
        len(cap) == 5000
        and attempted == 5000
        and len(m) >= 180
        and review_complete
        and evidence_ok
    )
    state = "SUBMISSION READY" if ready else "DRAFT - COMPLETION GATES PENDING"
    save_json(
        "outputs/readiness.json",
        {
            "ready": ready,
            "input": len(cap),
            "attempted": attempted,
            "usable_embeddings": len(m),
            "manual_audit_complete": review_complete,
            "qualitative_evidence_complete": evidence_ok,
        },
    )
    pdf = path("reports/findings.pdf")
    pdf.parent.mkdir(exist_ok=True)
    cv = canvas.Canvas(str(pdf), pagesize=(612, 792))
    cv.setTitle("Visual map of 5,000 websites")
    style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=HexColor("#26394a"),
    )

    def text(value, y, size=9):
        s = ParagraphStyle("s", parent=style, fontSize=size, leading=size + 3)
        p = Paragraph(escape(str(value)).replace("\n", "<br/>"), s)
        _, h = p.wrap(516, 700)

        if y - h < 48:
            raise RuntimeError(
                "Report text exceeds page budget. Shorten research notes or reduce figures."
            )
        p.drawOn(cv, 48, y - h)
        return y - h - 9

    def header(page, title):
        cv.setFillColor(HexColor("#15394c"))
        cv.rect(0, 722, 612, 70, fill=1, stroke=0)
        cv.setFillColorRGB(1, 1, 1)
        cv.setFont("Helvetica-Bold", 16 if len(title) > 43 else 18)
        cv.drawString(48, 754, title)
        cv.setFont("Helvetica", 8)
        cv.drawString(48, 735, state)
        cv.setFillColor(HexColor("#526675"))
        cv.setFont("Helvetica", 8)
        cv.drawString(48, 27, "Website Visual Map | local CPU research | seed 42")
        cv.drawRightString(564, 27, f"{page} / 4")

    def image(p, y, maxh):
        from PIL import Image

        with Image.open(require(p)) as im:
            w, h = im.size
        scale = min(516 / w, maxh / h)
        cv.drawImage(
            ImageReader(str(path(p))),
            48,
            y - h * scale,
            width=w * scale,
            height=h * scale,
        )
        return y - h * scale - 10

    header(1, "Coverage and the visual map")
    y = text(
        f"{attempted:,}/{len(cap):,} domains attempted; {captured:,} screenshots captured; {len(m):,} nonblank screenshots embedded. Capture success includes error pages, CAPTCHAs and login screens.",
        704,
        11,
    )
    y = text(
        f"Protocol: Chromium, 1280 x 800 viewport, no scrolling or consent interaction. Pretrained {MODEL} / {PRETRAINED}; CPU inference, unit-normalized vectors. OpenCLIP uses a resized center crop: peripheral layout and small text may be lost.",
        y,
    )
    y = image("outputs/visual_map.png", y, 320)
    y = text(
        f"{summary['projection']} is exploratory: nearby points suggest relationships but do not establish them. Clusters use KMeans selected from fixed candidates by cosine silhouette (k={summary['k']}, score={summary['silhouette_cosine']:.3f}).",
        y,
    )
    y = text(
        "Selection and capture bias: only successfully captured, nonblank images enter the map. Tier denominators include failed visits. These observations describe one browser, network, location and capture time, not a population-wide causal effect of popularity.",
        y,
    )
    text(
        "Reproduce: python run_pipeline.py. See README for installation, capture settings, manual audit, source references, hashes and submission gates.",
        y,
    )
    cv.showPage()
    header(2, "1. Do popular websites look different?")
    y = text(
        "Measured within-tier cosine similarity, with 300 site-level bootstrap resamples. The intervals describe uncertainty within the captured sample; coverage differences can confound tier comparisons.",
        704,
    )
    y = image("outputs/tier_comparison.png", y, 245)
    for r in tiers.to_dict("records"):
        value = r["within_tier_mean_cosine"]
        desc = f"{float(value):.3f}" if value != "" else "insufficient sample"
        y = text(
            f"{r['tier']}: {r['captured']}/{r['input_count']} captured, {r['embedded']} embedded; mean within-tier cosine {desc}.",
            y,
        )
    valid = tiers[pd.to_numeric(tiers.within_tier_mean_cosine, errors="coerce").notna()]
    if len(valid) > 1:
        hi = valid.loc[pd.to_numeric(valid.within_tier_mean_cosine).idxmax()]
        lo = valid.loc[pd.to_numeric(valid.within_tier_mean_cosine).idxmin()]
        y = text(
            f"Finding 1: {hi.tier} has the highest observed within-tier similarity and {lo.tier} the lowest (difference {float(hi.within_tier_mean_cosine) - float(lo.within_tier_mean_cosine):.3f}). This measures visual homogeneity, not aesthetic quality or a monotonic popularity effect.",
            y,
        )
    category_tiers = read("outputs/tier_category_analysis.csv")
    content = category_tiers[category_tiers.predicted_label == "content"]
    sensitivity = "; ".join(
        f"{r.tier}: n={r.count}, mean={float(r.within_group_mean_cosine):.3f}"
        for r in content.itertuples()
        if str(r.within_group_mean_cosine) not in ("", "nan")
    )
    y = text(
        "Sensitivity check, predicted content pages only: "
        + (sensitivity or "insufficient sample")
        + ". These unverified labels do not remove selection bias.",
        y,
    )
    y = image("outputs/tier_samples.png", y, min(160, max(50, y - 65)))
    cv.showPage()
    header(3, "2. Which websites look alike?")
    near = read("outputs/near_duplicates.csv")
    outs = read("outputs/outliers.csv")
    y = text(
        f"Finding 2: {len(near)} unique retrieved pairs have cosine at least 0.98. Reviewed examples below include identical captured content and shared access-error templates. The threshold flags candidates; it does not prove common ownership or similar intended website design.",
        704,
    )
    y = image("outputs/neighbor_pairs.png", y, 310)
    y = text(
        notes.get("lookalikes", {}).get(
            "text",
            "Qualitative review pending. Inspect neighbor_pairs.png and record screenshot IDs in data/research_notes.json.",
        ),
        y,
    )
    y = text(
        f"Outlier evidence: {outs.iloc[0].domain} has the lowest nearest-neighbor similarity ({float(outs.iloc[0].cosine_similarity):.3f}). Outliers mean unusual relative to this sample, not unusual on the whole web.",
        y,
    )
    image("outputs/outliers.png", y, min(190, max(60, y - 65)))
    cv.showPage()
    header(4, "3-4. Capture outcomes and embedding failures")
    y = text(
        "Finding 3: dominant visible page categories predicted by zero-shot CLIP. Categories compete and may overlap in reality; scores and margins are not calibrated confidence probabilities.",
        704,
    )
    counts = z.predicted_label.value_counts()
    y = text(
        "; ".join(f"{k}: {v} ({v / len(z):.1%})" for k, v in counts.items())
        + ". Denominator: embedded screenshots, not all domains.",
        y,
    )
    y = image("outputs/category_counts.png", y, 125)
    accuracy = (
        f"{audit['accuracy']:.1%}"
        if audit["accuracy"] is not None
        else "not yet measured"
    )
    reviewers = ", ".join(audit.get("reviewers", [])) or "pending"
    scored = audit.get("scored", audit["reviewed"])
    precision = "; ".join(
        f"{name} {values['precision']:.0%} ({round(values['precision'] * values['predicted'])}/{values['predicted']})"
        for name, values in audit["per_class"].items()
        if values["precision"] is not None
    )
    y = text(
        f"Visual audit by {reviewers}: {audit['reviewed']}/{audit['sample_size']} reviewed; {scored} non-uncertain labels; agreement {accuracy}. Precision: {precision}. Cookie and for-sale predictions were unreliable in this sample. Raw category counts do not establish true prevalence.",
        y,
    )
    y = text(
        "Finding 4: "
        + notes.get("embedding_failure", {}).get(
            "text",
            "Manual failure-case review is pending. Inspect disagreements and nearest neighbors; record a concrete misleading pair or category error in research_notes.json.",
        ),
        y,
    )
    y = image("outputs/failure_examples.png", y, 90)
    y = text(
        "Limitations: English text prompts, natural-image pretraining, center cropping and small text can obscure website semantics. Access blocks reflect collection outcomes. No CAPTCHA solving or login was attempted. Recaptures can differ because websites and network conditions change.",
        y,
    )
    y = text(
        "References: Playwright Python screenshots and Page API (playwright.dev/python/docs); OpenCLIP model and preprocessing (github.com/mlfoundations/open_clip); UMAP documentation (umap-learn.readthedocs.io). Model prompts, package versions, data hashes and settings are archived with the outputs.",
        y,
    )
    text(
        "Completion gates: see outputs/readiness.json; status shown in the page header.",
        y,
        8,
    )
    cv.save()
    print(f"Wrote 4-page {pdf}: {state}")


if __name__ == "__main__":
    main(report)
