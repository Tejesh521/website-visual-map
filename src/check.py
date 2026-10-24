"""Validate submission invariants without re-running inference or collection."""

import argparse
from pypdf import PdfReader
from common import *


def check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    a = parser.parse_args()
    m, x = load_embeddings()
    source = read("data/validated_domains.csv")
    c = read("metadata/capture_manifest.csv")
    q = read("metadata/quality.csv")
    errors = []
    v = json.loads(require("metadata/validation_summary.json").read_text())
    if v["input_sha256"] != digest(require("data/domains.csv")) or v[
        "validated_sha256"
    ] != digest(require("data/validated_domains.csv")):
        errors.append("Input CSV hashes differ from the validated collection.")
    protocol = json.loads(require("metadata/quality_protocol.json").read_text())
    if protocol["capture_manifest_sha256"] != digest(
        require("metadata/capture_manifest.csv")
    ):
        errors.append("Quality results are stale relative to captures.")
    if (
        set(source.site_id) != set(c.site_id)
        or len(c) != len(source)
        or c.site_id.duplicated().any()
    ):
        errors.append("Capture manifest does not map one-to-one to input.")
    usable = q[q.usable.astype(str).str.lower() == "true"]
    if set(usable.site_id) != set(m.site_id):
        errors.append("Usable images and embedding manifest differ.")
    for r in m.to_dict("records"):
        if (
            not path(r["screenshot"]).exists()
            or digest(path(r["screenshot"])) != r["sha256"]
        ):
            errors.append(f"Image integrity failed: {r['domain']}")
    if not np.array_equal(m.embedding_row.to_numpy(), np.arange(len(m))):
        errors.append("Embedding row order mismatch.")
    if len(PdfReader(require("reports/findings.pdf")).pages) != 4:
        errors.append("Report is not exactly four pages.")
    verify_analysis(
        [
            "outputs/nearest_neighbors.csv",
            "outputs/tier_analysis.csv",
            "outputs/zero_shot_labels.csv",
            "outputs/projection.csv",
        ]
    )
    readiness = json.loads(require("outputs/readiness.json").read_text())
    result = {
        "integrity_pass": not errors,
        "errors": errors,
        "submission_ready": readiness["ready"],
        "input_count": len(source),
        "embedding_count": len(m),
    }
    save_json("outputs/submission_check.json", result)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    if a.strict and not readiness["ready"]:
        raise RuntimeError(
            "Integrity passes but submission gates remain pending. See outputs/readiness.json."
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(check)
