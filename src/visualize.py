import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from common import *


def sheet(rows, filename, columns=4, thumb=(300, 188)):
    rows = list(rows)
    if not rows:
        return
    w, h = thumb
    canvas = Image.new(
        "RGB",
        (columns * w, ((len(rows) + columns - 1) // columns) * (h + 42)),
        "#eff2f5",
    )
    draw = ImageDraw.Draw(canvas)
    for i, r in enumerate(rows):
        xx = i % columns * w
        yy = i // columns * (h + 42)
        with Image.open(require(r["screenshot"])) as im:
            im = im.convert("RGB")
            im.thumbnail((w - 8, h))
            canvas.paste(im, (xx + 4, yy))
        draw.text((xx + 5, yy + h + 2), str(r.get("domain", ""))[:42], fill="#182438")
        draw.text(
            (xx + 5, yy + h + 17),
            str(r.get("caption", r.get("site_id", "")))[:45],
            fill="#43566a",
        )
    canvas.save(path(filename))


def visualize():
    m, x = load_embeddings()
    proj = read("outputs/projection.csv")
    c = read("outputs/clusters.csv")
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
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, g in proj.groupby("tier"):
        ax.scatter(g.x, g.y, s=9, alpha=0.6, label=f"{tier} (n={len(g)})")
    ax.legend(fontsize=8)
    ax.set(
        title="Website screenshot embeddings",
        xlabel="Projection axis 1",
        ylabel="Projection axis 2",
    )
    fig.tight_layout()
    fig.savefig(path("outputs/visual_map.png"))
    plt.close(fig)
    rows = c[c.representative.astype(str).str.lower() == "true"].to_dict("records")
    for r in rows:
        r["caption"] = "cluster " + str(r["cluster"])
    sheet(rows, "outputs/contact_sheet.png")
    tier_rows = []
    for tier, g in z.groupby("tier"):
        candidates = g[g.predicted_label == "content"]
        if len(candidates) < 2:
            candidates = g
        selected = candidates.sample(
            n=min(2, len(candidates)), random_state=SEED
        ).to_dict("records")
        for r in selected:
            r["caption"] = tier + " | predicted " + r["predicted_label"]
        tier_rows.extend(selected)
    sheet(tier_rows, "outputs/tier_samples.png", columns=5)
    sample = z.sample(n=min(48, len(z)), random_state=SEED).to_dict("records")
    for r in sample:
        r["caption"] = "predicted: " + r["predicted_label"]
    sheet(sample, "outputs/category_samples.png")
    manual = read("outputs/audit_sample.csv").to_dict("records")
    sheet(manual, "outputs/manual_audit_sheet.png")
    nn = read("outputs/nearest_neighbors.csv")
    nearest = nn[nn["rank"] == 1].sort_values("cosine_similarity", ascending=False)
    seen = set()
    pairs = []
    notes_path = path("data/research_notes.json")
    evidence = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    ids = evidence.get("lookalikes", {}).get("site_ids", [])[:2]
    if len(ids) == 2 and set(ids).issubset(set(m.site_id)):
        ix = [m.index[m.site_id == sid][0] for sid in ids]
        similarity = float(x[ix[0]] @ x[ix[1]])
        for sid in ids:
            item = m[m.site_id == sid].iloc[0].to_dict()
            item["caption"] = f"Reviewed pair | cosine {similarity:.3f}"
            pairs.append(item)
            seen.add(sid)
    for r in nearest.to_dict("records"):
        if r["site_id"] in seen or r["neighbor_id"] in seen:
            continue
        for sid in [r["site_id"], r["neighbor_id"]]:
            item = m[m.site_id == sid].iloc[0].to_dict()
            item["caption"] = f"cosine {r['cosine_similarity']:.3f}"
            pairs.append(item)
            seen.add(sid)
        if len(pairs) >= 12:
            break
    sheet(pairs, "outputs/neighbor_pairs.png", columns=4)
    outs = read("outputs/outliers.csv")
    sheet(
        m[m.site_id.isin(outs.site_id)].head(12).to_dict("records"),
        "outputs/outliers.png",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pos = np.arange(len(tiers))
    axes[0].bar(pos, tiers.capture_rate)
    axes[0].set_xticks(pos, tiers.tier, rotation=30, ha="right")
    axes[0].set(
        ylabel="Captured / input", ylim=(0, 1), title="Capture coverage by tier"
    )
    means = pd.to_numeric(tiers.within_tier_mean_cosine, errors="coerce")
    low = pd.to_numeric(tiers.bootstrap_low, errors="coerce")
    high = pd.to_numeric(tiers.bootstrap_high, errors="coerce")
    axes[1].errorbar(
        pos,
        means,
        yerr=[np.maximum(0, means - low), np.maximum(0, high - means)],
        fmt="o",
        capsize=4,
    )
    axes[1].set_xticks(pos, tiers.tier, rotation=30, ha="right")
    axes[1].set(
        ylabel="Mean pairwise cosine",
        title="Within-tier similarity (site bootstrap 95% CI)",
    )
    fig.tight_layout()
    fig.savefig(path("outputs/tier_comparison.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    z.predicted_label.value_counts().plot.bar(ax=ax, color="#287a91")
    ax.set(
        title="Raw zero-shot capture outcomes (model predictions)",
        ylabel="Screenshots",
    )
    fig.tight_layout()
    fig.savefig(path("outputs/category_counts.png"))
    plt.close(fig)
    notes_file = path("data/research_notes.json")
    if notes_file.exists():
        notes = json.loads(notes_file.read_text())
        ids = notes.get("embedding_failure", {}).get("site_ids", [])
        sheet(
            m[m.site_id.isin(ids)].to_dict("records"),
            "outputs/failure_examples.png",
            columns=3,
        )
    save_json(
        "outputs/figure_sources.json",
        {
            name: digest(require(name))
            for name in [
                "outputs/projection.csv",
                "outputs/clusters.csv",
                "outputs/tier_analysis.csv",
                "outputs/zero_shot_labels.csv",
                "outputs/nearest_neighbors.csv",
            ]
        },
    )
    print("Figures and audit contact sheet generated.")


if __name__ == "__main__":
    main(visualize)
