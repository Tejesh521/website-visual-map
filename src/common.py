from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib, json, os, platform, importlib.metadata
import numpy as np
import pandas as pd
from config import ROOT, SEED, MODEL, PRETRAINED


def path(p):
    return ROOT / p


def require(p):
    p = path(p)
    if not p.exists():
        raise RuntimeError(
            f"Missing {p.relative_to(ROOT)}. Run its preceding pipeline stage first."
        )
    return p


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(p, obj):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)


def save_csv(p, df):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    if p.parent == path("outputs") and path("embeddings/embeddings.npy").exists():
        save_json(str(p.relative_to(ROOT)) + ".provenance.json", analysis_signature())


def read(p):
    return pd.read_csv(require(p), keep_default_na=False)


def load_embeddings():
    m = read("embeddings/embedding_manifest.csv")
    meta = json.loads(require("embeddings/model.json").read_text())
    x = np.load(require("embeddings/embeddings.npy"))
    if digest(require("metadata/quality.csv")) != meta["quality_sha256"]:
        raise RuntimeError("Quality results changed. Rerun embed before analysis.")
    for key, filename in [
        ("array_sha256", "embeddings/embeddings.npy"),
        ("manifest_sha256", "embeddings/embedding_manifest.csv"),
    ]:
        if meta.get(key) and meta[key] != digest(require(filename)):
            raise RuntimeError(
                "Embedding artifact integrity failed. Rerun embed after archiving corrupted outputs."
            )
    if len(x) != len(m) or len(x) < 3 or not np.isfinite(x).all():
        raise RuntimeError(
            "Need at least 3 finite embeddings aligned with the manifest."
        )
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
        raise RuntimeError("Embeddings must be unit normalized.")
    return m, x


def model():
    import torch, open_clip
    from config import THREADS

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    from huggingface_hub import hf_hub_download
    from config import HF_REPOSITORY, HF_REVISION, WEIGHT_FILENAME, WEIGHT_SHA256

    checkpoint = hf_hub_download(
        HF_REPOSITORY, filename=WEIGHT_FILENAME, revision=HF_REVISION
    )
    if digest(checkpoint) != WEIGHT_SHA256:
        raise RuntimeError("Downloaded model weight checksum mismatch.")
    cfg = open_clip.get_pretrained_cfg(MODEL, PRETRAINED)
    net, _, preprocess = open_clip.create_model_and_transforms(
        MODEL,
        pretrained=checkpoint,
        device="cpu",
        image_mean=cfg["mean"],
        image_std=cfg["std"],
        image_interpolation=cfg["interpolation"],
        image_resize_mode=cfg["resize_mode"],
    )
    save_json(
        "metadata/model_weights.json",
        {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "filename": WEIGHT_FILENAME,
            "sha256": WEIGHT_SHA256,
        },
    )
    net.eval()
    return net, preprocess, open_clip.get_tokenizer(MODEL)


def versions():
    packages = [
        "numpy",
        "pandas",
        "Pillow",
        "playwright",
        "torch",
        "torchvision",
        "open_clip_torch",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "reportlab",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "packages": {p: importlib.metadata.version(p) for p in packages},
    }


def main(fn):
    try:
        fn()
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def analysis_signature():
    return {
        name: digest(require(name))
        for name in [
            "metadata/quality.csv",
            "embeddings/embedding_manifest.csv",
            "embeddings/embeddings.npy",
            "embeddings/model.json",
        ]
    }


def verify_analysis(names):
    expected = analysis_signature()
    for name in names:
        p = require(name + ".provenance.json")
        if json.loads(p.read_text()) != expected:
            raise RuntimeError(
                f"{name} is stale. Rerun analysis stages from similarity."
            )
