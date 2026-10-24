from common import *


def pair_sample(x, rng, n=10000):
    if len(x) < 2:
        return np.array([])
    a = rng.integers(len(x), size=n)
    b = rng.integers(len(x) - 1, size=n)
    b = b + (b >= a)
    return np.einsum("ij,ij->i", x[a], x[b])


def tier_analysis():
    m, x = load_embeddings()
    captures = read("metadata/capture_manifest.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for tier, g in captures.groupby("tier", sort=True):
        ix = np.flatnonzero(m.tier.values == tier)
        z = x[ix]
        values = pair_sample(z, rng)
        # Site-level bootstrap retains dependence among similarities sharing a website.
        estimates = []
        if len(z) > 1:
            for _ in range(300):
                sample = z[rng.integers(len(z), size=len(z))]
                n = len(sample)
                estimates.append(
                    float((np.square(sample.sum(axis=0)).sum() - n) / (n * (n - 1)))
                )
            n = len(z)
            mean = float((np.square(z.sum(axis=0)).sum() - n) / (n * (n - 1)))
            lo, hi = np.quantile(estimates, [0.025, 0.975])
        else:
            mean = lo = hi = np.nan
        rows.append(
            {
                "tier": tier,
                "input_count": len(g),
                "attempted": int(g.status.isin(["success", "failed"]).sum()),
                "captured": int((g.status == "success").sum()),
                "embedded": len(z),
                "capture_rate": float((g.status == "success").mean()),
                "within_tier_mean_cosine": mean,
                "bootstrap_low": lo,
                "bootstrap_high": hi,
                "sampled_pair_std": float(values.std()) if len(values) else np.nan,
            }
        )
    save_csv("outputs/tier_analysis.csv", pd.DataFrame(rows))
    cross = []
    for ta in sorted(m.tier.unique()):
        for tb in sorted(m.tier.unique()):
            if ta >= tb:
                continue
            a = x[m.tier.values == ta]
            b = x[m.tier.values == tb]
            cross.append(
                {
                    "tier_a": ta,
                    "tier_b": tb,
                    "mean_cosine": float(a.mean(axis=0) @ b.mean(axis=0)),
                    "n_a": len(a),
                    "n_b": len(b),
                }
            )
    save_csv(
        "outputs/cross_tier.csv",
        pd.DataFrame(cross, columns=["tier_a", "tier_b", "mean_cosine", "n_a", "n_b"]),
    )


if __name__ == "__main__":
    main(tier_analysis)
