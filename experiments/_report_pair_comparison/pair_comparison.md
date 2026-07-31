**Same defense, two pairs.** Identical attack, encoder, model, budget and defense; only the attacked class pair differs.

- **benefit** = PTA without the defense minus PTA with it: what the audit actually buys.
- **lift** = defended PTA minus that pair's clean-label floor (0.001 for truck→deer, 0.061 for dog→cat): what survives the audit. Raw PTA is not comparable across pairs, because a clean model already confuses dogs with cats and never confuses trucks with deer.

Blank truck→deer defended cells below $\tau = 1.0$ are deliberate, not missing data: the undefended attack there is already at the clean-label floor (PTA $\leq$ 0.017), and applying a defense to an attack that does not work measures nothing.

| tau | t→d undef | t→d def | benefit | lift | recall | d→c undef | d→c def | benefit | lift | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.996 | 0.111 | 0.885 | 0.110 | 0.913 | 0.950 | 0.562 | 0.388 | 0.501 | 0.677 |
| 0.90 | 0.017 | -- | -- | -- | 0.805 | 0.646 | 0.364 | 0.282 | 0.303 | 0.522 |
| 0.80 | 0.003 | -- | -- | -- | 0.703 | 0.325 | 0.289 | 0.036 | 0.228 | 0.356 |
| 0.70 | 0.002 | -- | -- | -- | 0.531 | 0.219 | 0.188 | 0.031 | 0.127 | 0.362 |
| 0.60 | 0.001 | -- | -- | -- | 0.311 | 0.177 | 0.138 | 0.039 | 0.077 | 0.236 |
| 0.50 | 0.002 | -- | -- | -- | 0.000 | 0.134 | 0.118 | 0.016 | 0.057 | 0.000 |
