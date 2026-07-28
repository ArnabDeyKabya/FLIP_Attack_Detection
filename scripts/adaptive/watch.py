"""
Live epoch-by-epoch view of the running sweep. Read-only; safe to start, stop
and restart at any time without touching the sweep.

    python scripts/adaptive/watch.py                 follow from now on
    python scripts/adaptive/watch.py --from-start    replay the whole log first

The trainer draws a single tqdm bar spanning all 200 epochs and refreshes its
postfix once per epoch, so one printed line here is exactly one finished epoch:

    ep  96/200   train 97.34   CTA 89.87   PTA 99.60   loss 0.0003012  lr 0.01

    CTA = clean test accuracy      (does the model still look normal?)
    PTA = poisoned test accuracy   (does the backdoor fire? — the headline)

tqdm redraws in place with carriage returns, so plain `Get-Content -Wait` shows
one unreadable mega-line; this splits on \\r and prints only what changed.
"""

import argparse
import functools
import re
import time
from pathlib import Path


# A live view is worthless if it sits in a buffer. Python block-buffers stdout
# whenever it is not a TTY, which is exactly what happens the moment anyone
# pipes this into `more`, `head`, or a file.
print = functools.partial(print, flush=True)  # noqa: A001


# Trailing tqdm record: "... 47%|#### | 4700416/10000000 [57:13<1:00:48, ...]"
# followed by the postfix tqdm sorts alphabetically: acc, acc0, acc1, loss, lr.
BAR_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
POSTFIX_RE = re.compile(
    r"acc=([\d.]+),\s*acc0=([\d.]+),\s*acc1=([\d.]+),\s*"
    r"loss=([\d.eE+-]+),\s*lr=([\d.eE+-]+)"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log",
        default="experiments/_report_adaptive_cifar_1xs/sweep_none.log",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--from-start", action="store_true",
                   help="Replay existing log content before following.")
    p.add_argument("--poll", type=float, default=1.0,
                   help="Seconds between reads.")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    log = root / args.log

    print(f"watching {log}")
    # Plain ASCII: the Windows console codepage mangles non-ASCII punctuation.
    print("Ctrl+C stops watching only - the sweep keeps running.\n")

    while not log.exists():
        print("waiting for log file...")
        time.sleep(5)

    handle = open(log, "rb")
    if not args.from_start:
        handle.seek(0, 2)

    pending = ""
    last_postfix = None
    last_current = None
    run_no = 0

    try:
        while True:
            blob = handle.read()
            if not blob:
                time.sleep(args.poll)
                continue

            pending += blob.decode("utf-8", errors="replace")
            # tqdm separates in-place redraws with \r; keep the final partial
            # record in `pending` so a half-written line is never parsed.
            records = re.split(r"[\r\n]", pending)
            pending = records.pop()

            for rec in records:
                postfix = POSTFIX_RE.search(rec)
                if not postfix:
                    continue
                bar = BAR_RE.search(rec)
                if not bar:
                    continue

                current = int(bar.group(2))
                total = int(bar.group(3))
                values = postfix.groups()

                # A restart of the sample counter means the sweep moved on to
                # the next experiment.
                if last_current is not None and current < last_current:
                    last_postfix = None
                    print()
                if last_current is None or current < last_current:
                    run_no += 1
                    print(f"===== run {run_no} =====")
                last_current = current

                if values == last_postfix:
                    continue
                last_postfix = values

                train, cta, pta, loss, lr = values
                epoch = max(1, round(current / total * args.epochs))
                print(
                    f"ep {epoch:3d}/{args.epochs}   train {train:>6}   "
                    f"CTA {cta:>6}   PTA {pta:>6}   "
                    f"loss {loss:>10}  lr {lr}"
                )
    except KeyboardInterrupt:
        print("\nstopped watching (sweep still running)")
    finally:
        handle.close()


if __name__ == "__main__":
    main()
