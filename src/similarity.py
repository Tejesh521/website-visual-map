from sklearn.neighbors import NearestNeighbors
from common import *


def similarity():
    m, x = load_embeddings()
    k = min(6, len(x))
    distances, indices = (
        NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        .fit(x)
        .kneighbors(x)
    )
    rows = []
    for i, (ds, ix) in enumerate(zip(distances, indices)):
        rank = 0
        for d, j in zip(ds, ix):
            if i == j:
                continue
            if rank >= 5:
                break
            rank += 1
            rows.append(
                {
                    "site_id": m.iloc[i].site_id,
                    "domain": m.iloc[i].domain,
                    "neighbor_id": m.iloc[j].site_id,
                    "neighbor": m.iloc[j].domain,
                    "rank": rank,
                    "cosine_similarity": float(1 - d),
                    "same_image": m.iloc[i].sha256 == m.iloc[j].sha256,
                }
            )
    out = pd.DataFrame(rows)
    save_csv("outputs/nearest_neighbors.csv", out)
    near = out[out["cosine_similarity"] >= 0.98].copy()
    near["_pair"] = [
        tuple(sorted([a, b])) for a, b in zip(near.site_id, near.neighbor_id)
    ]
    near = near.drop_duplicates("_pair").drop(columns="_pair")
    save_csv("outputs/near_duplicates.csv", near)
    outliers = out[out["rank"] == 1].sort_values("cosine_similarity").head(24)
    save_csv("outputs/outliers.csv", outliers)
    print(
        f"Similarity: {len(out)} neighbor links; {len(near)} high-similarity pairs (not verified duplicates)."
    )


if __name__ == "__main__":
    main(similarity)
