# Does the audit still detect a defense-aware attacker?

Attacker optimises against k=20, threshold 0.5. Defender columns vary k to test whether simply auditing at a different k recovers detection.

`recall` is the fraction of actually-placed poisons the defender removes at its operating threshold — the number that decides whether the backdoor survives.

## Threshold 0.5

| tau | poisons placed | recall @ k=5 | recall @ k=20 | recall @ k=100 | recall @ k=500 | AUPRC @ attacker k | clean removed @ attacker k |
|---|---|---|---|---|---|---|
| 1.00 | 1500 | 0.643 | 0.677 | 0.785 | 0.860 | 0.348 | 1712 |
| 0.90 | 1500 | 0.530 | 0.522 | 0.626 | 0.619 | 0.286 | 1660 |
| 0.80 | 1500 | 0.357 | 0.356 | 0.430 | 0.406 | 0.251 | 1165 |
| 0.70 | 1297 | 0.391 | 0.362 | 0.383 | 0.410 | 0.270 | 910 |
| 0.60 | 846 | 0.301 | 0.236 | 0.267 | 0.317 | 0.165 | 958 |
| 0.50 | 556 | 0.201 | 0.000 | 0.142 | 0.221 | 0.088 | 1044 |
| 0.40 | 336 | 0.128 | 0.000 | 0.074 | 0.158 | 0.045 | 1175 |
| 0.30 | 180 | 0.061 | 0.000 | 0.006 | 0.078 | 0.017 | 1312 |
| 0.20 | 107 | 0.019 | 0.000 | 0.000 | 0.065 | 0.008 | 1371 |
| 0.10 | 48 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 1428 |

