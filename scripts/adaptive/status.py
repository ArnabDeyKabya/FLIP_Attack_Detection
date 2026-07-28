"""
One-screen progress report for the adaptive sweep. Safe to run any time.

Reads only files on disk, so it never disturbs the running sweep and works
even if the sweep was started from another terminal or has since died.

    python scripts/adaptive/status.py

Shows per-run state (done / running / pending), the finished runs' CTA and PTA
so the frontier can be read as it fills in, and an ETA from the live tqdm line
in the log.
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


TQDM_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+)"
)


def parse_hms(text: str):
    """'1:02:03' or '12:34' -> timedelta; None when tqdm prints '?'."""
    if "?" in text:
        return None
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


def tail_tqdm(log_path: Path, tail_bytes: int = 200_000):
    """Last tqdm progress record in the log, or None."""
    if not log_path.exists():
        return None
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - tail_bytes))
        blob = f.read().decode("utf-8", errors="replace")
    matches = TQDM_RE.findall(blob.replace("\r", "\n"))
    if not matches:
        return None
    pct, cur, total, elapsed, remaining = matches[-1]
    return {
        "pct": int(pct),
        "current": int(cur),
        "total": int(total),
        "elapsed": parse_hms(elapsed),
        "remaining": parse_hms(remaining),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="experiments/_report_adaptive_cifar_1xs/sweep_none.log")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    p.add_argument("--modes", nargs="+", default=["none"])
    # nargs="*" so `--control-budgets` with no values means "no controls",
    # which is the reduced scope after tau=0.90 made them redundant.
    p.add_argument("--control-budgets", type=int, nargs="*",
                   default=[150, 300, 500, 1000])
    p.add_argument("--name-prefix", default="adaptive_cifar_1xs")
    p.add_argument("--control-prefix", default="flipbudget_cifar_1xs")
    p.add_argument("--clean-control-name", default="cleanlabel_cifar_1xs_none")
    p.add_argument("--no-clean-control", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]

    planned = [
        (f"{args.name_prefix}_tau{f'{t:.3f}'.replace('.', 'p')}_{m}",
         f"tau={t:.2f} {m}")
        for m in args.modes for t in args.taus
    ] + [
        (f"{args.control_prefix}_n{n}_none", f"control n={n}")
        for n in args.control_budgets
    ] + ([] if args.no_clean_control else
         [(args.clean_control_name, "clean-label control")])

    done, pending, running = [], [], None
    print(f"{'run':<38} {'state':<9} {'CTA':>7} {'PTA':>7}")
    print("-" * 64)
    for name, label in planned:
        exp = root / "experiments" / name
        summary = exp / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            done.append(name)
            print(f"{label:<38} {'done':<9} {s['cta']:>7.4f} {s['pta']:>7.4f}")
        elif running is None and (exp / "config.toml").exists() and not pending:
            running = name
            print(f"{label:<38} {'RUNNING':<9} {'':>7} {'':>7}")
        else:
            pending.append(name)
            print(f"{label:<38} {'pending':<9} {'':>7} {'':>7}")

    print("-" * 64)
    print(f"{len(done)}/{len(planned)} complete")

    prog = tail_tqdm(root / args.log)
    if prog and running:
        epochs_total = 200
        epoch = round(prog["current"] / prog["total"] * epochs_total)
        line = (f"current run: {prog['pct']}%  (epoch ~{epoch}/{epochs_total})")
        if prog["remaining"]:
            line += f"  ~{prog['remaining']} left"
        print(line)

        if prog["elapsed"] and prog["pct"] > 0:
            per_run = prog["elapsed"] / max(prog["pct"], 1) * 100
            left = (prog["remaining"] or timedelta(0)) + per_run * len(pending)
            eta = datetime.now() + left
            print(f"sweep ETA:   {eta:%a %d %b %H:%M}  "
                  f"(~{left.total_seconds() / 3600:.1f} h at "
                  f"~{per_run.total_seconds() / 60:.0f} min/run)")
    elif not pending and not running:
        print("sweep finished — next: "
              "python scripts/adaptive/aggregate_frontier.py")
    else:
        print("no live progress in log (sweep may not be running)")


if __name__ == "__main__":
    main()
