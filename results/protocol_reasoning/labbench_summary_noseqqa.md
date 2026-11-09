| model | macro acc [95% CI] | ProtocolQA | Cloning | macro precision | macro coverage | filtered | trunc | tok/q | $ |
|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | 0.414 [0.358, 0.471] | 0.662 | 0.167 | 0.604 | 0.628 | 0.000 | 0.195 | 9459 | 1.11 |
| gpt-6-luna | 0.614 [0.530, 0.694] | 0.653 | 0.576 | 0.746 | 0.823 | 0.000 | 0.000 | 943 | 0.23 |
| haiku-5.5 | 0.630 [0.548, 0.710] | 0.639 | 0.621 | 0.729 | 0.867 | 0.007 | 0.007 | 1398 | 0.33 |

- glm-5.3-flash - gpt-6-luna: -0.200 macro-acc, 95% CI [-0.278, -0.121]  (significant)
- glm-5.3-flash - haiku-5.5: -0.216 macro-acc, 95% CI [-0.299, -0.134]  (significant)
- gpt-6-luna - haiku-5.5: -0.016 macro-acc, 95% CI [-0.108, +0.072]  (not significant)