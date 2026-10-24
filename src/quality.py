import numpy as np
import io
from PIL import Image
from common import *


def quality():
    snapshot = require("metadata/capture_manifest.csv").read_bytes()
    df = pd.read_csv(io.BytesIO(snapshot), keep_default_na=False)
    rows = []
    for r in df.to_dict("records"):
        out = {
            "site_id": r["site_id"],
            "domain": r["domain"],
            "tier": r["tier"],
            "screenshot": r.get("screenshot", ""),
            "usable": False,
            "reason": "capture_failed",
            "sha256": "",
            "pixel_std": 0,
            "entropy": 0,
        }
        if r.get("status") == "success":
            try:
                p = require(r["screenshot"])
                if digest(p) != r["sha256"]:
                    raise ValueError("Screenshot hash mismatch")
                with Image.open(p) as im:
                    im.load()
                    gray = np.asarray(im.convert("L").resize((256, 160)))
                    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
                    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
                    std = float(gray.std())
                    usable = (
                        im.width >= 640
                        and im.height >= 400
                        and std >= 3
                        and entropy >= 0.3
                    )
                    out.update(
                        usable=usable,
                        reason="pass" if usable else "blank_or_small",
                        sha256=digest(p),
                        pixel_std=std,
                        entropy=entropy,
                    )
            except (OSError, ValueError, RuntimeError) as e:
                out["reason"] = str(e)
        rows.append(out)
    save_csv("metadata/quality.csv", pd.DataFrame(rows))
    save_json(
        "metadata/quality_protocol.json",
        {
            "capture_manifest_sha256": hashlib.sha256(snapshot).hexdigest(),
            "minimum_std": 3,
            "minimum_entropy": 0.3,
            "minimum_size": [640, 400],
        },
    )
    print(
        f"Quality: {sum(r['usable'] for r in rows)} usable, {len(rows)} total. Pass means nonblank, not intended homepage."
    )


if __name__ == "__main__":
    main(quality)
