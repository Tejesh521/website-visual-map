import torch
from common import *
from config import CATEGORIES


def zero_shot():
    m, x = load_embeddings()
    net, _, tokenizer = model()
    names = list(CATEGORIES)
    text = []
    with torch.inference_mode():
        for prompts in CATEGORIES.values():
            v = net.encode_text(tokenizer(prompts), normalize=True).mean(dim=0)
            text.append(v / v.norm())
    sims = x @ torch.stack(text).cpu().numpy().T
    order = np.argsort(-sims, axis=1)
    out = m[["site_id", "domain", "tier", "screenshot", "sha256"]].copy()
    out["predicted_label"] = [names[i] for i in order[:, 0]]
    out["score"] = sims[np.arange(len(x)), order[:, 0]]
    out["margin"] = (
        sims[np.arange(len(x)), order[:, 0]] - sims[np.arange(len(x)), order[:, 1]]
    )
    for j, name in enumerate(names):
        out["cosine_" + name] = sims[:, j]
    save_csv("outputs/zero_shot_labels.csv", out)
    save_json(
        "outputs/zero_shot_protocol.json",
        {
            "categories": CATEGORIES,
            "interpretation": "Mutually exclusive dominant visible page type; cosine scores are not calibrated probabilities.",
        },
    )
    # Fixed random audit sample, independent of predicted classes, supports honest descriptive validation.
    sample = (
        out.sample(n=min(80, len(out)), random_state=SEED).sort_values("site_id").copy()
    )
    sample["manual_label"] = ""
    sample["notes"] = ""
    sample["reviewer"] = ""
    target = path("data/manual_labels.csv")
    if target.exists():
        manual = read("data/manual_labels.csv")
        if "sha256" not in manual:
            manual["sha256"] = ""
        current = dict(zip(sample.site_id, sample.sha256))
        for idx, row in manual.iterrows():
            if row.site_id in current and row.sha256 != current[row.site_id]:
                manual.loc[idx, ["manual_label", "notes", "reviewer"]] = ""
                manual.loc[idx, "sha256"] = current[row.site_id]
        missing = sample[~sample.site_id.isin(manual.site_id)]
        manual = pd.concat([manual, missing[manual.columns]], ignore_index=True)
    else:
        manual = sample[
            [
                "site_id",
                "domain",
                "screenshot",
                "sha256",
                "manual_label",
                "notes",
                "reviewer",
            ]
        ]
    save_csv("data/manual_labels.csv", manual)
    save_csv("outputs/audit_sample.csv", sample[["site_id", "domain", "screenshot"]])
    manual = manual[manual.site_id.isin(sample.site_id)].copy()
    if manual.site_id.duplicated().any():
        raise ValueError("Duplicate site IDs in manual labels.")
    bad = set(manual.manual_label) - set(names) - {"", "uncertain"}
    if bad:
        raise ValueError(f"Unknown manual labels: {bad}")
    reviewed = manual[
        manual.manual_label.isin(names) & manual.reviewer.astype(str).str.strip().ne("")
    ].merge(out, on="site_id", suffixes=("_manual", ""))
    metrics = {
        "sample_size": len(manual),
        "reviewed": int(
            (
                manual.manual_label.isin(names + ["uncertain"])
                & manual.reviewer.astype(str).str.strip().ne("")
            ).sum()
        ),
        "scored": len(reviewed),
        "reviewers": sorted(set(manual.loc[manual.manual_label != "", "reviewer"])),
        "accuracy": None,
        "per_class": {},
        "note": "Random audit sample; manual labels describe capture outcomes. Uncertain labels excluded.",
    }
    if len(reviewed):
        metrics["accuracy"] = float(
            (reviewed.manual_label == reviewed.predicted_label).mean()
        )
        for name in names:
            tp = int(
                (
                    (reviewed.manual_label == name) & (reviewed.predicted_label == name)
                ).sum()
            )
            predicted = int((reviewed.predicted_label == name).sum())
            actual = int((reviewed.manual_label == name).sum())
            metrics["per_class"][name] = {
                "actual": actual,
                "predicted": predicted,
                "precision": tp / predicted if predicted else None,
                "recall": tp / actual if actual else None,
            }
        save_csv(
            "outputs/manual_disagreements.csv",
            reviewed[reviewed.manual_label != reviewed.predicted_label],
        )
    breakdown = []
    for (tier, label), g in out.groupby(["tier", "predicted_label"]):
        ix = g.index.to_numpy()
        n = len(ix)
        mean = (
            float((np.square(x[ix].sum(axis=0)).sum() - n) / (n * (n - 1)))
            if n > 1
            else None
        )
        breakdown.append(
            {
                "tier": tier,
                "predicted_label": label,
                "count": n,
                "within_group_mean_cosine": mean,
            }
        )
    save_csv("outputs/tier_category_analysis.csv", pd.DataFrame(breakdown))
    metrics["manual_labels_sha256"] = digest(require("data/manual_labels.csv"))
    metrics["predictions_sha256"] = digest(require("outputs/zero_shot_labels.csv"))
    save_json("outputs/classification_audit.json", metrics)
    print(
        f"Zero-shot labels generated. Audit: {metrics['reviewed']}/{len(manual)} reviewed; {len(reviewed)} scored."
    )


if __name__ == "__main__":
    main(zero_shot)
