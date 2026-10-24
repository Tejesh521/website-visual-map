"""Bounded, resumable homepage visits. JSONL journal is the durable source of truth."""

import argparse, asyncio, time
from filelock import FileLock, Timeout as LockTimeout
from datetime import datetime, timezone
from common import *
from config import VIEWPORT
from playwright.async_api import async_playwright


def journal_rows():
    p = path("metadata/capture_attempts.jsonl")
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a torn last write
    return rows


def materialize(rows, df):
    if not rows:
        return
    latest = pd.DataFrame(rows).drop_duplicates("site_id", keep="last")
    save_csv(
        "metadata/capture_manifest.csv",
        df.merge(
            latest.drop(columns=["domain", "tier", "crux_origin"], errors="ignore"),
            on="site_id",
            how="left",
        ),
    )


async def _capture(a):
    df = read("data/validated_domains.csv")
    validation = json.loads(require("metadata/validation_summary.json").read_text())
    if validation["input_sha256"] != digest(require("data/domains.csv")):
        raise RuntimeError("Input changed; rerun validate.")
    if validation["validated_sha256"] != digest(require("data/validated_domains.csv")):
        raise RuntimeError("Validated input changed. Rerun validate.")
    protocol = {
        "viewport": VIEWPORT,
        "timeout_ms": a.timeout * 1000,
        "settle_ms": a.settle * 1000,
        "retries": a.retries,
        "input_sha256": validation["validated_sha256"],
        "browser": "chromium",
        "https_fallback": "http after failed origin",
    }
    pp = path("metadata/capture_protocol.json")
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise RuntimeError(
            "Capture protocol changed. Use original settings or archive metadata and screenshots for a new run."
        )
    save_json("metadata/capture_protocol.json", protocol)
    rows = journal_rows()
    latest = {r["site_id"]: r for r in rows}
    pending = []
    for r in df.to_dict("records"):
        old = latest.get(r["site_id"])
        if (
            old
            and old["status"] == "success"
            and path(old["screenshot"]).exists()
            and digest(path(old["screenshot"])) == old["sha256"]
        ):
            continue
        if old and not a.retry_failed and old["status"] != "success":
            continue
        pending.append(r)
    # Round-robin tiers so interrupted runs are less dominated by the first tier.
    groups = {t: [r for r in pending if r["tier"] == t] for t in df.tier.unique()}
    pending = [
        g[i]
        for i in range(max([len(g) for g in groups.values()] + [0]))
        for g in groups.values()
        if i < len(g)
    ]
    if a.limit:
        pending = pending[: a.limit]
    queue = asyncio.Queue()
    for r in pending:
        queue.put_nowait(r)
    path("screenshots").mkdir(exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        save_json("metadata/browser.json", {"version": browser.version})

        async def worker():
            while not queue.empty():
                try:
                    r = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                history = []
                result = None
                start = time.monotonic()
                for attempt in range(a.retries + 1):
                    url = (
                        r["crux_origin"]
                        if attempt == 0
                        else "http://" + urlsplit_host(r["crux_origin"])
                    )
                    ctx = None
                    out = {
                        **r,
                        "status": "failed",
                        "screenshot": "",
                        "sha256": "",
                        "http_status": None,
                        "final_url": "",
                        "title": "",
                        "error": "",
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "attempt": attempt + 1,
                    }
                    t = time.monotonic()
                    try:
                        ctx = await browser.new_context(
                            viewport=VIEWPORT,
                            device_scale_factor=1,
                            ignore_https_errors=False,
                            accept_downloads=False,
                        )
                        page = await ctx.new_page()
                        page.set_default_timeout(a.timeout * 1000)

                        async def visit():
                            response = await page.goto(
                                url,
                                wait_until="domcontentloaded",
                                timeout=a.timeout * 1000,
                            )
                            out["http_status"] = response.status if response else None
                            out["navigation_ms"] = round((time.monotonic() - t) * 1000)
                            await page.wait_for_timeout(a.settle * 1000)
                            out["final_url"] = page.url
                            out["title"] = (await page.title())[:300]
                            filename = f"screenshots/{r['site_id']}.png"
                            await page.screenshot(
                                path=str(path(filename)),
                                full_page=False,
                                animations="disabled",
                                timeout=a.timeout * 1000,
                            )
                            out.update(
                                status="success",
                                screenshot=filename,
                                sha256=digest(path(filename)),
                            )

                        await asyncio.wait_for(
                            visit(), timeout=a.timeout * 2 + a.settle + 5
                        )
                    except Exception as e:
                        out["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                    finally:
                        if ctx:
                            try:
                                await asyncio.wait_for(ctx.close(), 5)
                            except Exception:
                                pass
                    history.append(
                        {
                            "url": url,
                            "error": out["error"],
                            "http_status": out["http_status"],
                            "duration_ms": round((time.monotonic() - t) * 1000),
                        }
                    )
                    result = out
                    if out["status"] == "success":
                        break
                result["elapsed_ms"] = round((time.monotonic() - start) * 1000)
                result["attempt_history"] = json.dumps(history)
                rows.append(result)
                with open(path("metadata/capture_attempts.jsonl"), "a") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                if len(rows) % 25 == 0:
                    materialize(rows, df)
                    print(
                        f"Completed {len(rows)} records; last {r['domain']}: {result['status']}",
                        flush=True,
                    )
                queue.task_done()

        try:
            await asyncio.gather(*(worker() for _ in range(a.concurrency)))
        finally:
            materialize(rows, df)
            await browser.close()
    print(
        f"Capture finished: {len(pending)} visits this run; {len(set(r['site_id'] for r in rows))}/{len(df)} domains attempted."
    )


async def capture(a):
    path("metadata").mkdir(exist_ok=True)
    try:
        with FileLock(str(path("metadata/capture.lock")), timeout=0):
            await _capture(a)
    except LockTimeout:
        raise RuntimeError("Another capture process holds the lock.")


def urlsplit_host(url):
    from urllib.parse import urlsplit

    return urlsplit(url).netloc


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument(
        "--timeout", type=float, help="Seconds; reuse saved protocol or default to 20"
    )
    p.add_argument("--settle", type=float)
    p.add_argument("--retries", type=int)
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    protocol_path = path("metadata/capture_protocol.json")
    previous = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    if a.timeout is None:
        a.timeout = previous.get("timeout_ms", 20000) / 1000
    if a.settle is None:
        a.settle = previous.get("settle_ms", 1000) / 1000
    if a.retries is None:
        a.retries = previous.get("retries", 1)
    if a.limit is not None and a.limit < 1:
        raise ValueError("--limit must be positive.")
    if not 1 <= a.concurrency <= 64 or a.timeout <= 0 or a.settle < 0 or a.retries < 0:
        raise ValueError("Invalid capture limits.")
    asyncio.run(capture(a))


if __name__ == "__main__":
    main(cli)
