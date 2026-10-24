"""All stages execute in dependency order; capture and embedding safely resume."""

import argparse, subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    "validate",
    "capture",
    "quality",
    "embed",
    "similarity",
    "cluster",
    "tier_analysis",
    "zero_shot",
    "visualize",
    "report",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-stage", choices=STAGES, default="validate")
    p.add_argument("--to-stage", choices=STAGES, default="report")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--timeout", type=float)
    p.add_argument("--settle", type=float)
    p.add_argument("--retries", type=int)
    a = p.parse_args()
    start = STAGES.index(a.from_stage)
    end = STAGES.index(a.to_stage)
    if start > end:
        p.error("from-stage must precede to-stage")
    for stage in STAGES[start : end + 1]:
        command = [sys.executable, str(ROOT / "src" / f"{stage}.py")]
        if stage == "capture":
            command += ["--concurrency", str(a.concurrency)]
            for name in ["timeout", "settle", "retries"]:
                value = getattr(a, name)
                if value is not None:
                    command += ["--" + name, str(value)]
            if a.limit:
                command += ["--limit", str(a.limit)]
        print(f"\n=== {stage} ===", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise SystemExit(
                f"Stage {stage} failed (exit {result.returncode}). Fix the error and resume with --from-stage {stage}."
            )
    if end == len(STAGES) - 1:
        ready = json.loads((ROOT / "outputs/readiness.json").read_text())
        print("Submission gates:", ready)


if __name__ == "__main__":
    main()
