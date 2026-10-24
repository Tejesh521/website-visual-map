import torch
from PIL import Image
from common import *
from config import BATCH_SIZE, MODEL, PRETRAINED, HF_REVISION, WEIGHT_SHA256


def embed():
    q = read("metadata/quality.csv")
    q = q[q.usable.astype(str).str.lower() == "true"].copy().reset_index(drop=True)
    if len(q) < 3:
        raise RuntimeError(
            "Need at least 3 usable screenshots. Run capture and quality."
        )
    signature = {
        "quality_sha256": digest(require("metadata/quality.csv")),
        "model": MODEL,
        "pretrained": PRETRAINED,
        "preprocess": "OpenCLIP default resize + center crop; RGB; L2 normalization",
        "device": "cpu",
        "weight_revision": HF_REVISION,
        "weight_sha256": WEIGHT_SHA256,
    }
    signature["encoder_versions"] = {
        k: versions()["packages"][k]
        for k in ["torch", "torchvision", "open_clip_torch", "Pillow"]
    }
    # Verify images even when reusing a complete embedding cache.
    for r in q.to_dict("records"):
        if digest(require(r["screenshot"])) != r["sha256"]:
            raise RuntimeError(f"Image changed: {r['domain']}; rerun quality.")
    old = path("embeddings/model.json")
    if (
        old.exists()
        and all(json.loads(old.read_text()).get(k) == v for k, v in signature.items())
        and path("embeddings/embeddings.npy").exists()
    ):
        load_embeddings()
        print("Embedding cache verified.")
        return
    # Reuse individual images across growing collections, keyed by exact image hash.
    for r in q.to_dict("records"):
        if digest(require(r["screenshot"])) != r["sha256"]:
            raise RuntimeError(f"Image changed: {r['domain']}; rerun quality.")
    cache = {}
    if (
        old.exists()
        and path("embeddings/embeddings.npy").exists()
        and path("embeddings/embedding_manifest.csv").exists()
    ):
        previous = json.loads(old.read_text())
        if all(
            previous.get(k) == signature[k]
            for k in ["model", "pretrained", "preprocess", "device"]
        ):
            files_match = all(
                previous.get(key, digest(require(filename)))
                == digest(require(filename))
                for key, filename in [
                    ("array_sha256", "embeddings/embeddings.npy"),
                    ("manifest_sha256", "embeddings/embedding_manifest.csv"),
                ]
            )
            previous_versions = previous.get(
                "encoder_versions",
                {
                    k: previous.get("versions", {}).get("packages", {}).get(k)
                    for k in signature["encoder_versions"]
                },
            )
            if not files_match or previous_versions != signature["encoder_versions"]:
                raise RuntimeError(
                    "Embedding cache integrity or library versions changed. Archive embeddings and rerun embed."
                )
            old_m = read("embeddings/embedding_manifest.csv")
            old_x = np.load(path("embeddings/embeddings.npy"))
            if (
                len(old_m) == len(old_x)
                and np.isfinite(old_x).all()
                and np.allclose(np.linalg.norm(old_x, axis=1), 1, atol=1e-4)
            ):
                cache = {r.sha256: old_x[i] for i, r in old_m.iterrows()}
    pending = q[~q.sha256.isin(cache)].drop_duplicates("sha256").reset_index(drop=True)
    print(
        f"Reusing {len(q) - len(q[~q.sha256.isin(cache)])} vectors; encoding {len(pending)} distinct new images.",
        flush=True,
    )
    if len(pending):
        net, preprocess, _ = model()
        for i in range(0, len(pending), BATCH_SIZE):
            images = []
            batch = pending.iloc[i : i + BATCH_SIZE]
            for r in batch.to_dict("records"):
                with Image.open(path(r["screenshot"])) as im:
                    images.append(preprocess(im.convert("RGB")))
            with torch.inference_mode():
                v = (
                    net.encode_image(torch.stack(images), normalize=True)
                    .cpu()
                    .float()
                    .numpy()
                )
            for sha, vector in zip(batch.sha256, v):
                cache[sha] = vector
            if i % 80 == 0:
                print(
                    f"Embedded {min(i + BATCH_SIZE, len(pending))}/{len(pending)} new images",
                    flush=True,
                )
    vectors = [cache[sha] for sha in q.sha256]
    path("embeddings").mkdir(exist_ok=True)
    x = np.stack(vectors)
    with open(path("embeddings/embeddings.npy.tmp"), "wb") as f:
        np.save(f, x)
    os.replace(path("embeddings/embeddings.npy.tmp"), path("embeddings/embeddings.npy"))
    q.insert(0, "embedding_row", np.arange(len(q)))
    save_csv("embeddings/embedding_manifest.csv", q)
    signature.update(
        dimensions=x.shape[1],
        count=len(x),
        versions=versions(),
        array_sha256=digest(path("embeddings/embeddings.npy")),
        manifest_sha256=digest(path("embeddings/embedding_manifest.csv")),
    )
    save_json("embeddings/model.json", signature)


if __name__ == "__main__":
    main(embed)
