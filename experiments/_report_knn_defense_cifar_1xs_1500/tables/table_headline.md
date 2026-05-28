## Headline: Deep k-NN defense across k

AUROC / AUPRC measured on the full training set scores; flag P / R
evaluated at the configured auto threshold (default 0.5).

| k | AUROC | AUPRC | CTA none | CTA remove | PTA none | PTA remove | flag P | flag R | best F1 | best MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.848 | 0.246 | 0.898 | 0.911 | 0.997 | 0.470 | 0.320 | 0.745 | 0.448 | 0.466 |
| 5 | 0.974 | 0.515 | 0.897 | 0.916 | 1.000 | 0.386 | 0.405 | 0.846 | 0.592 | 0.581 |
| 10 | 0.983 | 0.572 | 0.895 | 0.913 | 0.998 | 0.219 | 0.464 | 0.846 | 0.618 | 0.613 |
