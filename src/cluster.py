from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from common import *


def cluster():
    m, x = load_embeddings()
    candidates = []
    for k in sorted(set([2, 3, 5, 8, 12])):
        if k >= len(x):
            continue
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            x, labels, metric="cosine", sample_size=min(1500, len(x)), random_state=SEED
        )
        candidates.append((float(score), k, labels))
    if not candidates:
        raise RuntimeError("No distinct clusters; inspect duplicate or blank captures.")
    score, k, labels = max(candidates, key=lambda v: v[0])
    out = m[["site_id", "domain", "tier", "screenshot"]].copy()
    out["cluster"] = labels
    # Representative = cosine medoid approximation (closest to normalized centroid).
    out["representative"] = False
    for label in set(labels):
        ix = np.flatnonzero(labels == label)
        c = x[ix].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        out.loc[ix[np.argmax(x[ix] @ c)], "representative"] = True
    save_csv("outputs/clusters.csv", out)
    if len(x) >= 10:
        import umap

        coords = umap.UMAP(
            n_neighbors=min(15, len(x) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=SEED,
            n_jobs=1,
        ).fit_transform(x)
        method = "UMAP"
    else:
        coords = PCA(n_components=2, random_state=SEED).fit_transform(x)
        method = "PCA (small sample fallback)"
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    save_csv("outputs/projection.csv", out)
    save_json(
        "outputs/cluster_summary.json",
        {
            "k": k,
            "silhouette_cosine": score,
            "candidates": [{"k": v[1], "score": v[0]} for v in candidates],
            "projection": method,
            "seed": SEED,
            "note": "Exploratory KMeans on unit vectors; projection distances do not establish similarity.",
        },
    )
    print(f"Clusters: k={k}, silhouette={score:.3f}; {method}")


if __name__ == "__main__":
    main(cluster)
