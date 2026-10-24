import sys, json, argparse, asyncio, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common, validate, quality, capture, similarity


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    for d in ["data", "metadata", "screenshots", "embeddings", "outputs"]:
        (tmp_path / d).mkdir()
    return tmp_path


def test_validate_idn_ports_and_tiers(root):
    pd.DataFrame(
        [
            {
                "domain": "xn--80ahdzi4a2eb.xn--p1ai",
                "tier": "low",
                "crux_origin": "https://xn--80ahdzi4a2eb.xn--p1ai",
            },
            {
                "domain": "example.org",
                "tier": "high",
                "crux_origin": "https://sub.example.org:446",
            },
        ]
    ).to_csv(root / "data/domains.csv", index=False)
    validate.validate()
    out = pd.read_csv(root / "data/validated_domains.csv")
    assert len(out) == 2 and out.iloc[1].crux_origin.endswith(":446")
    assert out.site_id.is_unique


@pytest.mark.parametrize(
    "rows",
    [
        [{"domain": "example.org", "tier": "a", "crux_origin": "https://evil.org"}],
        [
            {"domain": "example.org", "tier": "a"},
            {"domain": "example.org", "tier": "b"},
        ],
        [{"domain": "../bad", "tier": "a"}],
    ],
)
def test_reject_invalid_input(root, rows):
    pd.DataFrame(rows).to_csv(root / "data/domains.csv", index=False)
    with pytest.raises(ValueError):
        validate.validate()


def test_quality_blank_and_hash_tampering(root):
    rows = []
    for i in range(3):
        p = root / f"screenshots/{i}.png"
        array = (
            np.zeros((800, 1280, 3), dtype=np.uint8)
            if i == 0
            else np.random.default_rng(i).integers(
                0, 255, (800, 1280, 3), dtype=np.uint8
            )
        )
        Image.fromarray(array).save(p)
        rows.append(
            {
                "site_id": str(i),
                "domain": f"{i}.test",
                "tier": "x",
                "status": "success",
                "screenshot": f"screenshots/{i}.png",
                "sha256": common.digest(p) if i != 2 else "bad",
            }
        )
    pd.DataFrame(rows).to_csv(root / "metadata/capture_manifest.csv", index=False)
    quality.quality()
    q = pd.read_csv(root / "metadata/quality.csv")
    assert list(q.usable) == [False, True, False]


@pytest.mark.parametrize("status", [200, 503])
def test_real_browser_capture_resume(root, status):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b'<title>Fixture</title><body style="background:navy;color:white"><h1>Website fixture</h1></body>'
            )

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    row = {
        "domain": "fixture.test",
        "tier": "one",
        "crux_origin": f"http://127.0.0.1:{server.server_port}",
        "site_id": "fixture",
    }
    pd.DataFrame([row]).to_csv(root / "data/validated_domains.csv", index=False)
    (root / "data/domains.csv").write_text("fixture")
    common.save_json(
        "metadata/validation_summary.json",
        {
            "input_sha256": common.digest(root / "data/domains.csv"),
            "validated_sha256": common.digest(root / "data/validated_domains.csv"),
        },
    )
    a = argparse.Namespace(
        timeout=3, settle=0, retries=0, concurrency=1, limit=None, retry_failed=False
    )
    try:
        asyncio.run(capture.capture(a))
        asyncio.run(capture.capture(a))
        records = capture.journal_rows()
        assert len(records) == 1
        assert (
            records[0]["status"] == "success"
            and records[0]["http_status"] == status
            and records[0]["title"] == "Fixture"
        )
        assert Image.open(root / records[0]["screenshot"]).size == (1280, 800)
    finally:
        server.shutdown()


def test_similarity_excludes_self_with_ties(root):
    m = pd.DataFrame(
        {
            "site_id": ["a", "b", "c"],
            "domain": ["a.org", "b.org", "c.org"],
            "sha256": ["1", "1", "3"],
        }
    )
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    common.save_csv("metadata/quality.csv", pd.DataFrame({"x": [1]}))
    common.save_csv("embeddings/embedding_manifest.csv", m)
    np.save(root / "embeddings/embeddings.npy", x)
    common.save_json(
        "embeddings/model.json",
        {"quality_sha256": common.digest(root / "metadata/quality.csv")},
    )
    similarity.similarity()
    n = pd.read_csv(root / "outputs/nearest_neighbors.csv")
    assert (n.site_id != n.neighbor_id).all()
    assert n[(n.site_id == "a") & (n.neighbor_id == "b")].iloc[0].cosine_similarity == 1
    common.save_csv("metadata/quality.csv", pd.DataFrame({"x": [2]}))
    with pytest.raises(RuntimeError, match="changed"):
        common.load_embeddings()


def test_many_identical_vectors_still_limit_neighbors(root):
    ids = list("abcdefgh")
    m = pd.DataFrame(
        {
            "site_id": ids,
            "domain": [i + ".org" for i in ids],
            "sha256": ["identical"] * 8,
        }
    )
    x = np.tile([1.0, 0.0], (8, 1)).astype(np.float32)
    common.save_csv("metadata/quality.csv", pd.DataFrame({"x": [1]}))
    common.save_csv("embeddings/embedding_manifest.csv", m)
    np.save(root / "embeddings/embeddings.npy", x)
    common.save_json(
        "embeddings/model.json",
        {"quality_sha256": common.digest(root / "metadata/quality.csv")},
    )
    similarity.similarity()
    n = pd.read_csv(root / "outputs/nearest_neighbors.csv")
    assert (n.site_id != n.neighbor_id).all()
    assert n.groupby("site_id").size().eq(5).all()
    pairs = pd.read_csv(root / "outputs/near_duplicates.csv")
    expected = {tuple(sorted([a, b])) for a, b in zip(n.site_id, n.neighbor_id)}
    actual = {tuple(sorted([a, b])) for a, b in zip(pairs.site_id, pairs.neighbor_id)}
    assert actual == expected
