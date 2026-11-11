| model | budget | ProtocolQA acc | Cloning acc | macro acc | macro cov | macro prec | macro F1 | trunc (PQA / Cloning) | mean tokens/q |
|---|---|---|---|---|---|---|---|---|---|
| haiku-5.5 | 4k (fixed) | 0.67 | 0.45 | 0.56 | 0.79 | 0.70 | 0.74 | 0.00 / 0.24 | 1879 |
| haiku-5.5 | 12k (fixed) | 0.67 | 0.58 | 0.62 | 0.88 | 0.71 | 0.78 | 0.00 / 0.03 | 2096 |
| haiku-5.5 | as needed (64k cap) | 0.67 | 0.61 | 0.64 | 0.90 | 0.71 | 0.79 | 0.00 / 0.00 | 2256 |
| gpt-6-luna | 4k (fixed) | 0.66 | 0.42 | 0.54 | 0.69 | 0.78 | 0.73 | 0.00 / 0.24 | 1442 |
| gpt-6-luna | 12k (fixed) | 0.66 | 0.58 | 0.62 | 0.82 | 0.76 | 0.78 | 0.00 / 0.00 | 1603 |
| gpt-6-luna | as needed (64k cap) | 0.66 | 0.58 | 0.62 | 0.82 | 0.76 | 0.78 | 0.00 / 0.00 | 1603 |
| glm-5.3-flash | 4k (fixed) | 0.51 | 0.06 | 0.28 | 0.35 | 0.89 | 0.41 | 0.34 / 0.94 | 3003 |
| glm-5.3-flash | 12k (fixed) | 0.60 | 0.12 | 0.36 | 0.46 | 0.80 | 0.52 | 0.23 / 0.79 | 7590 |
| glm-5.3-flash | as needed (64k cap) | 0.70 | 0.33 | 0.52 | 0.74 | 0.68 | 0.71 | 0.06 / 0.36 | 18952 |

Tokens actually used per question, as-needed run (completion incl. reasoning):

| model | mean | median | p90 | max | rows |
|---|---|---|---|---|---|
| haiku-5.5 | 1432 | 783 | 3428 | 22545 | 141 |
| gpt-6-luna | 907 | 298 | 2683 | 11925 | 141 |
| glm-5.3-flash | 12384 | 1988 | 64000 | 64000 | 141 |

Rows counted as cut off at 64k because the request had not finished when the run was stopped: haiku-5.5 0, gpt-6-luna 0, glm-5.3-flash 4