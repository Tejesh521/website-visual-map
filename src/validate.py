import argparse, re
from urllib.parse import urlsplit
from common import *


def validate(input_file="data/domains.csv"):
    p = require(input_file)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if not {"domain", "tier"}.issubset(df):
        raise ValueError("CSV requires domain and tier columns.")
    if len(df) == 0:
        raise ValueError("CSV contains no rows.")
    rows = []
    seen = set()
    for i, r in df.iterrows():
        d = r.domain.strip().lower().rstrip(".")
        try:
            d = d.encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError(f"Invalid IDN at row {i + 2}")
        if not re.fullmatch(
            r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,63}|xn--[a-z0-9-]{2,59})",
            d,
        ):
            raise ValueError(f"Invalid domain at row {i + 2}: {d}")
        if d in seen:
            raise ValueError(f"Duplicate domain: {d}")
        seen.add(d)
        origin = r.get("crux_origin", "").strip() or "https://" + d
        u = urlsplit(origin)
        if (
            u.scheme not in ("https", "http")
            or u.username
            or u.password
            or not u.hostname
        ):
            raise ValueError(f"Invalid origin for {d}")
        host = u.hostname.lower().rstrip(".")
        if host != d and not host.endswith("." + d):
            raise ValueError(f"Origin host does not belong to {d}")
        if not r.tier.strip():
            raise ValueError(f"Missing tier for {d}")
        rows.append(
            {
                **r.to_dict(),
                "domain": d,
                "crux_origin": f"{u.scheme}://{host}"
                + (f":{u.port}" if u.port else ""),
                "site_id": hashlib.sha256(d.encode()).hexdigest()[:16],
            }
        )
    clean = pd.DataFrame(rows)
    save_csv("data/validated_domains.csv", clean)
    save_json(
        "metadata/validation_summary.json",
        {
            "rows": len(clean),
            "expected_rows": 5000,
            "matches_target": len(clean) == 5000,
            "input_sha256": digest(p),
            "validated_sha256": digest(path("data/validated_domains.csv")),
            "tiers": clean.tier.value_counts().to_dict(),
        },
    )
    print(
        f"Validated {len(clean)} unique domains; tiers: {clean.tier.value_counts().to_dict()}"
    )


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/domains.csv")
    a = p.parse_args()
    validate(a.input)


if __name__ == "__main__":
    main(cli)
