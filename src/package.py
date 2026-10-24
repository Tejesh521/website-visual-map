"""Archive a frozen research run and verify its SHA-256 inventory."""

import argparse, tarfile
from common import *


def inventory(include_images=False):
    files = []
    for pattern in [
        "src/*.py",
        "tests/*.py",
        "*.py",
        "*.txt",
        "*.md",
        "*.csv",
        ".gitignore",
        "data/*",
        "metadata/*.json",
        "metadata/*.jsonl",
        "metadata/*.csv",
        "outputs/*",
        "reports/*.pdf",
    ]:
        files.extend(
            p
            for p in ROOT.glob(pattern)
            if p.is_file() and p.name != "artifact_hashes.json"
        )
    if include_images:
        files.extend(ROOT.glob("screenshots/*.png"))
        files.extend(ROOT.glob("embeddings/*"))
    return sorted(set(files))


def package():
    p = argparse.ArgumentParser()
    p.add_argument("--include-images", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    if a.verify:
        hashes = json.loads(require("metadata/artifact_hashes.json").read_text())[
            "sha256"
        ]
        failures = [
            f for f, h in hashes.items() if not path(f).exists() or digest(path(f)) != h
        ]
        if failures:
            raise RuntimeError("Hash verification failed: " + ", ".join(failures[:10]))
        print(f"Verified {len(hashes)} artifact hashes.")
        return
    files = inventory(a.include_images)
    save_json(
        "metadata/artifact_hashes.json",
        {
            "includes_images": a.include_images,
            "sha256": {str(f.relative_to(ROOT)): digest(f) for f in files},
        },
    )
    target = path("reports/reproduction_bundle.tar.gz")
    target.parent.mkdir(exist_ok=True)
    with tarfile.open(target, "w:gz", compresslevel=3) as tar:
        for f in files + [path("metadata/artifact_hashes.json")]:
            tar.add(f, arcname=str(f.relative_to(ROOT)))
    print(
        f"Wrote {target} ({target.stat().st_size / 1024**2:.1f} MB), images included: {a.include_images}"
    )


if __name__ == "__main__":
    main(package)
