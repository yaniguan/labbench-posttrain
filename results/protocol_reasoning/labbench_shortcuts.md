# LAB-Bench No-Model Shortcut Baselines

Counting rule: `'Insufficient information to answer the question'` is excluded from option-length comparisons; exact length ties split credit evenly (`1/k`). Position baseline uses `seed=0` shuffled choice order.

| family | n | chance (with refuse) | chance (excl. refuse) | longest (chars) | longest (words) | shortest (chars) | modal pos (seed 0) | always refuse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ProtocolQA | 108 | 0.188 | 0.232 | 0.332 | 0.321 | 0.179 | 0.241 (E) | 0.000 |
| CloningScenarios | 33 | 0.183 | 0.226 | 0.197 | 0.157 | 0.227 | 0.242 (D) | 0.000 |
| SeqQA | 600 | 0.198 | 0.247 | 0.260 | 0.247 | 0.177 | 0.215 (E) | 0.000 |
| LitQA2 | 199 | 0.194 | 0.244 | 0.257 | 0.223 | 0.228 | 0.211 (D) | 0.000 |
| DbQA | 520 | 0.200 | 0.250 | 0.235 | 0.250 | 0.282 | 0.217 (A) | 0.000 |
| SuppQA | 82 | 0.175 | 0.213 | 0.222 | 0.275 | 0.228 | 0.207 (E) | 0.000 |
| FigQA | 181 | 0.181 | 0.223 | 0.267 | 0.251 | 0.159 | 0.204 (D) | 0.000 |
| TableQA | 244 | 0.170 | 0.206 | 0.226 | 0.224 | 0.172 | 0.205 (C) | 0.000 |
| **macro (all 8)** | **1967** | **0.186** | **0.230** | **0.250** | **0.243** | **0.207** | **0.218** | **0.000** |
