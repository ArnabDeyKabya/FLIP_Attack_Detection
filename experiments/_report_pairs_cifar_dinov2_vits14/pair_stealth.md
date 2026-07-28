**Stealthy flip pool per class pair.** How many *source-class* images could be relabelled to the target and still score at or below the audit threshold. Computed from encoder neighbourhoods alone -- no attack, no training. A pair with a pool of zero cannot be attacked stealthily at all, which is why the defense appears total there.

| source -> target | mean audit score | pool @ tau<=0.9 | pool @ tau<=0.7 | pool @ tau<=0.5 |
|---|---|---|---|---|
| dog -> cat | 0.922 | 1164 | 527 | 265 |
| truck -> car | 0.973 | 425 | 164 | 83 |
| cat -> dog | 0.940 | 1223 | 314 | 82 |
| car -> truck | 0.974 | 402 | 143 | 72 |
| bird -> plane | 0.988 | 165 | 68 | 43 |
| deer -> frog | 0.989 | 150 | 79 | 42 |
| bird -> frog | 0.986 | 213 | 87 | 39 |
| cat -> frog | 0.982 | 310 | 104 | 39 |
| plane -> ship | 0.988 | 176 | 61 | 36 |
| bird -> deer | 0.984 | 267 | 88 | 36 |
| cat -> deer | 0.984 | 303 | 78 | 32 |
| deer -> horse | 0.987 | 216 | 58 | 30 |
| dog -> deer | 0.987 | 204 | 73 | 30 |
| horse -> deer | 0.989 | 174 | 61 | 29 |
| dog -> horse | 0.990 | 156 | 55 | 28 |
| ... | |  |  |  |
| truck -> deer  **<- yours** | 1.000 | 8 | 0 | 0 |
