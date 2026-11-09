| model | macro acc [95% CI] | ProtocolQA | Cloning | SeqQA | LitQA2 | DbQA | SuppQA | FigQA | TableQA | macro precision | macro coverage | filtered | trunc | tok/q | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | 0.416 [0.389, 0.443] | 0.676 | 0.182 | 0.335 | 0.211 | 0.181 | 0.232 | 0.635 | 0.873 | 0.661 | 0.590 | 0.000 | 0.441 | 7006 | 5.57 |
| gpt-6-luna | 0.596 [0.566, 0.626] | 0.657 | 0.576 | 0.722 | 0.412 | 0.458 | 0.268 | 0.757 | 0.918 | 0.687 | 0.860 | 0.000 | 0.000 | 1113 | 1.40 |
| haiku-5.5 | 0.595 [0.564, 0.623] | 0.667 | 0.576 | 0.847 | 0.372 | 0.394 | 0.220 | 0.790 | 0.893 | 0.689 | 0.848 | 0.012 | 0.006 | 1419 | 1.75 |

- glm-5.3-flash - gpt-6-luna: -0.180 macro-acc, 95% CI [-0.214, -0.146]  (significant)
- glm-5.3-flash - haiku-5.5: -0.179 macro-acc, 95% CI [-0.212, -0.146]  (significant)
- gpt-6-luna - haiku-5.5: +0.001 macro-acc, 95% CI [-0.029, +0.034]  (not significant)