# Does the audit still detect a defense-aware attacker?

Attacker optimises against k=20, threshold 0.5. Defender columns vary k to test whether simply auditing at a different k recovers detection.

`recall` is the fraction of actually-placed poisons the defender removes at its operating threshold — the number that decides whether the backdoor survives.

## Threshold 0.5

| tau | poisons placed | recall @ k=5 | recall @ k=20 | recall @ k=100 | recall @ k=500 | AUPRC @ attacker k | clean removed @ attacker k |
|---|---|---|---|---|---|---|
| 1.00 | 1500 | 0.846 | 0.913 | 0.969 | 0.999 | 0.614 | 1611 |
| 0.90 | 1500 | 0.759 | 0.805 | 0.873 | 0.894 | 0.489 | 1336 |
| 0.80 | 1263 | 0.669 | 0.703 | 0.749 | 0.765 | 0.401 | 1065 |
| 0.70 | 652 | 0.523 | 0.531 | 0.567 | 0.604 | 0.182 | 1153 |
| 0.60 | 399 | 0.406 | 0.311 | 0.383 | 0.456 | 0.089 | 1216 |
| 0.50 | 260 | 0.258 | 0.000 | 0.200 | 0.304 | 0.044 | 1276 |
| 0.40 | 171 | 0.170 | 0.000 | 0.099 | 0.193 | 0.023 | 1322 |
| 0.30 | 98 | 0.092 | 0.000 | 0.031 | 0.071 | 0.009 | 1393 |
| 0.20 | 58 | 0.017 | 0.000 | 0.000 | 0.017 | 0.004 | 1422 |
| 0.10 | 31 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 1446 |

