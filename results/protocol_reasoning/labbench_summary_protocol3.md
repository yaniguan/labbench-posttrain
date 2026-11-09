| model | macro acc [95% CI] | ProtocolQA | Cloning | SeqQA | macro precision | macro coverage | filtered | trunc | tok/q | $ |
|---|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | 0.384 [0.344, 0.422] | 0.662 | 0.167 | 0.324 | 0.696 | 0.541 | 0.000 | 0.535 | 9305 | 5.36 |
| gpt-6-luna | 0.652 [0.598, 0.704] | 0.653 | 0.576 | 0.727 | 0.774 | 0.841 | 0.000 | 0.000 | 1925 | 1.63 |
| haiku-5.5 | 0.697 [0.642, 0.752] | 0.639 | 0.621 | 0.830 | 0.780 | 0.892 | 0.006 | 0.022 | 2495 | 2.15 |

- glm-5.3-flash - gpt-6-luna: -0.267 macro-acc, 95% CI [-0.320, -0.216]  (significant)
- glm-5.3-flash - haiku-5.5: -0.312 macro-acc, 95% CI [-0.369, -0.254]  (significant)
- gpt-6-luna - haiku-5.5: -0.045 macro-acc, 95% CI [-0.101, +0.016]  (not significant)