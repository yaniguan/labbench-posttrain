# Biological protocol reasoning: LAB-Bench eval and BioProBench RL environments

## 1. Eval design

- Data: public LAB-Bench. The protocol subset is 741 questions in three families.
  - ProtocolQA (108): a written protocol plus a bad result, pick the fix. Only 24 distinct protocols, so the real sample size is closer to 24.
  - CloningScenarios (33): multi-step cloning design.
  - SeqQA (600): PCR, digest and ORF calculations.
- I also ran the other five families (1,226 questions) closed-book: LitQA2, DbQA, SuppQA, FigQA, TableQA.
- Setup: LAB-Bench zero-shot CoT prompt, temperature 0, with its "Insufficient information" option. Two answer orders. Output cap 12k tokens. GLM questions that hit the cap were re-run at 32k (ProtocolQA and Cloning only).
- Metrics: accuracy = correct / all. Coverage = answered / all (cut-off outputs count as unanswered). Precision = correct / answered. F1 = harmonic mean of precision and coverage. Macro = average over families. Chance is about 0.19.
- Limit: ProtocolQA is troubleshooting only. It has no planning items and about 2 counterfactual items.

## 2. Eval results

Trunc = share of outputs that hit the max_tokens cap.

| view | model | accuracy | coverage | precision | F1 | trunc |
|---|---|---|---|---|---|---|
| ProtocolQA (108) | Haiku 5.5 | 0.64 | 0.82 | 0.78 | 0.80 | 0.00 |
| | GPT 6 Luna | 0.65 | 0.84 | 0.78 | 0.81 | 0.00 |
| | GLM 5.3 flash | 0.66 | 0.91 | 0.73 | 0.81 | 0.08 |
| macro, 3 core families | Haiku 5.5 | 0.70 | 0.89 | 0.78 | 0.83 | 0.02 |
| | GPT 6 Luna | 0.65 | 0.84 | 0.77 | 0.81 | 0.00 |
| | GLM 5.3 flash | 0.38 | 0.54 | 0.70 | 0.58 | 0.54 |
| macro, no SeqQA | Haiku 5.5 | 0.63 | 0.87 | 0.73 | 0.79 | 0.01 |
| | GPT 6 Luna | 0.61 | 0.82 | 0.75 | 0.78 | 0.00 |
| | GLM 5.3 flash | 0.41 | 0.63 | 0.60 | 0.61 | 0.20 |
| macro, all 8 (one order) | Haiku 5.5 | 0.595 | 0.85 | 0.69 | 0.74 | 0.01 |
| | GPT 6 Luna | 0.596 | 0.86 | 0.69 | 0.75 | 0.00 |
| | GLM 5.3 flash | 0.42 | 0.59 | 0.66 | 0.61 | 0.44 |

- Haiku and GPT are not significantly different in any view. The gap is -4.5 points on the core subset, -1.6 without SeqQA, 0.0 over all 8. Haiku's core lead comes from SeqQA and from answering more questions, not from higher precision.
- On ProtocolQA all three tie at 0.64 to 0.66. The interval is about 7 to 10 points either side. When they answer, they are still wrong 22 to 27% of the time.
- GLM's low score is mostly truncation. 45 to 61% of its outputs were cut off at 12k on the text-heavy families. It also uses about 7k tokens per question, against 1.1k to 1.4k for the others.
- Haiku returned an empty content-filter response on 24 of 1,967 questions. Counted as wrong.
- LAB-Bench alone is a weak basis for picking a base model.

### Token budget

ProtocolQA and Cloning only (141 questions, one answer order). Three budgets: 4k and 12k fixed, and as needed (64k cap, applied to questions cut off at 12k). The 4k row is simulated from the 12k run: an output over 4,096 tokens would have been cut. Four GLM requests were still running when I stopped, so they count as cut off. GLM's as-needed row is a lower bound.

| model | budget | ProtocolQA acc | Cloning acc | macro acc | coverage | precision | F1 | trunc (PQA / Cloning) | tokens per question |
|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | 4k | 0.67 | 0.45 | 0.56 | 0.79 | 0.70 | 0.74 | 0.00 / 0.24 | 1,879 |
| | 12k | 0.67 | 0.58 | 0.62 | 0.88 | 0.71 | 0.78 | 0.00 / 0.03 | 2,096 |
| | as needed | 0.67 | 0.61 | 0.64 | 0.90 | 0.71 | 0.79 | 0.00 / 0.00 | 2,256 |
| GPT 6 Luna | 4k | 0.66 | 0.42 | 0.54 | 0.69 | 0.78 | 0.73 | 0.00 / 0.24 | 1,442 |
| | 12k | 0.66 | 0.58 | 0.62 | 0.82 | 0.76 | 0.78 | 0.00 / 0.00 | 1,603 |
| | as needed | 0.66 | 0.58 | 0.62 | 0.82 | 0.76 | 0.78 | 0.00 / 0.00 | 1,603 |
| GLM 5.3 flash | 4k | 0.51 | 0.06 | 0.28 | 0.35 | 0.89 | 0.41 | 0.34 / 0.94 | 3,003 |
| | 12k | 0.60 | 0.12 | 0.36 | 0.46 | 0.80 | 0.52 | 0.23 / 0.79 | 7,590 |
| | as needed | 0.70 | 0.33 | 0.52 | 0.74 | 0.68 | 0.71 | 0.06 / 0.36 | 18,952 |

- Haiku and GPT barely change above 4k. They only lose at 4k on Cloning.
- GLM gets better at every step. ProtocolQA goes 0.51, 0.60, 0.70. Cloning goes 0.06, 0.12, 0.33, and 36% of outputs still hit 64k.
- Median tokens are close (GLM 1,988, Haiku 783, GPT 298). But a tenth of GLM's questions use the full 64k, so its mean cost is 9 to 14 times higher.
- With budget as needed: Haiku 0.64, GPT 0.62, GLM 0.52 (macro accuracy). Haiku and GPT are within noise. ProtocolQA is a tie. GLM is far behind on Cloning.

## 3. RL environments

- I built one environment per BioProBench task: PQA, ORD, ERR, GEN.
- I skipped REA. Its train file is empty and it needs an LLM judge.
- Every reward is plain code, in [0, 1], and logs its parts. That is RLVR, no reward model and no judge.
- Some parts are exact checks (PQA correct, ORD exact order, ERR detect). Others compare to one reference by similarity (ORD tau, ERR repair, GEN). A valid answer can differ from the reference.
- The weights are my choice. I did not tune them.

| task (train size) | data | what it tests | reward and why |
|---|---|---|---|
| PQA (47k) | One step with a blank: parameter, reagent or operation, 5 choices | Quantities, reagents, operations | 0.8 x correct + 0.2 x calibration. The model gives a 0-100 confidence, scored with Brier. Confident misses lose most. |
| ORD (28k) | Shuffled steps or section titles (I keep 2 to 12) | Order and dependencies | 0.5 x exact + 0.5 x max(0, Kendall tau). Partial orders get credit. Random orders get about 0.2. |
| ERR (93k) | A step plus context. Half are correct, half have one injected error | Find the error, fix it, don't flag good steps | Correct step: 1 for "True". Corrupted step: 0.4 x detect + 0.6 x repair. Copying the bad step earns 0 repair. |
| GEN (107k) | A title and a reference step list (I keep 3 to 25 steps) | Write a full, ordered protocol with numbers | 0.6 x step F1 + 0.2 x order + 0.2 x numbers and units. Output over 3x the reference length is clipped. |

Baseline mean reward on 100 test prompts, greedy:

| env | Haiku 5.5 | GPT 6 Luna | GLM 5.3 flash |
|---|---|---|---|
| PQA | 0.67 | 0.73 | 0.64 |
| ORD | 0.66 | 0.70 | 0.60 |
| ERR | 0.60 | 0.55 | 0.44 |
| GEN | 0.22 | 0.19 | 0.22 |

- At 4 samples per prompt, 84 to 90% of PQA and GEN prompts have reward variance. Only 8 to 34% of ORD and ERR prompts do. Those two need bigger groups or filtering.
- Spurious rewards (Shao et al., 2026): random rewards raised MATH-500 by 21.4 points on Qwen2.5-Math-7B, against 29.1 for ground truth. They did not help Llama or OLMo. So a gain from training needs a control to compare with.
- Each env has `reward_mode`: true, random, format, incorrect (a plausible wrong label). Nothing was trained with them.
- `python -m bioenvs.audit` runs exploit policies. Always-True on ERR scores 0.47, a boilerplate GEN protocol 0.08. Tests pass.
- Limits: GEN scores against one reference, so valid but different protocols lose points. ERR repair is ambiguous when a number can't be inferred. No policy was trained.

## 4. Next steps

1. Base model: compare 2 or 3 open-weight models. Pick by headroom and reward variance, not raw score.
2. Data: BioProBench train. Split by source protocol. Check overlap with BioProBench test and LAB-Bench.
3. Evaluation, frozen before training: BioProBench test, LAB-Bench transfer, and probes for repair, counterfactuals and confident errors.
4. Real lab: replay logged runs first and check if the model suggests the fix the scientists used. Add blind expert ratings. Needs the lab's run logs.
5. Offline training: SFT, then rejection sampling, maybe DPO. Cheap, and shows how much is just format.
6. Online RL: GRPO first, then PPO. Same rollouts and GPU hours, same start, 3 seeds, KL penalty.
7. Controls: repeat runs with random, format and incorrect rewards on at least 2 model families. Report the gap to the best control.
8. Reward hacking: re-run the audit after every reward change. Keep a second verifier out of training. Stop if training reward rises and it falls. Watch length.
9. Gaming the benchmark: subtract shortcut baselines (on ProtocolQA the longest option is right 42% of the time, random is 23%). Keep the transfer set out of training. Test with shuffled options.

## 5. In this repository

The numbers above are copied from the completed run. They were not recomputed here. Summaries are in `results/protocol_reasoning/`. Raw per-question model outputs are not included.

- Harness: `labeval/`. Command: `python -m labeval.run`.
- Environments: `bioenvs/` (PQA, ORD, ERR, GEN). Commands: `python -m bioenvs.baseline` and `python -m bioenvs.audit`.
- Reward tests: `tests/test_envs.py`.
- GRPO wiring example (not run): `train/grpo_example.py`.
- LAB-Bench text: `data/labbench/`. License: CC-BY-SA-4.0 ([Future-House/LAB-Bench](https://github.com/Future-House/LAB-Bench)). `data/labbench/LICENSE-LAB-Bench` is that license. Canary strings stay in the jsonl files. Do not train on `data/labbench`. FigQA and TableQA images are not included. Clone the upstream repo and set `LABBENCH_REPO`.
- BioProBench samples: `data/bioprobench/`. License: CC BY-NC 4.0 ([YuyangSunshine/bioprobench](https://github.com/YuyangSunshine/bioprobench), also linked as bioprotocolbench). This data is not covered by the MIT code license.
- OpenRouter key: `OPENROUTER_API_KEY` in the environment or a gitignored `.env`. No key is stored in the repo.
