# labbench-posttrain

A small lab post-training project with two tracks, plus a protocol-reasoning eval and four reward environments:

- **Scientist:** learn scientific representations, align modalities with text, answer historical questions with evidence labels, and run SFT, DPO, reward modeling, GRPO and distillation.
- **Operator:** turn protocols into actions, learn state/action behavior and a world model, and score structured plans against simple physical constraints.
- **Protocol reasoning:** score public LAB-Bench questions, and train against four BioProBench tasks with code-checked rewards. See [Protocol reasoning](#protocol-reasoning) and [the write-up](docs/protocol_reasoning.md).

**GPU results: PENDING (A100 Colab).** CPU smoke checks verify code execution. They do not establish useful scientific accuracy or safe robot control.

See [the design mapped to code](docs/DESIGN.md), [dataset overview](docs/dataset_overview.md), [benchmark map](docs/benchmark_map.md), [assay quality](docs/assay_quality.md), and [CPU smoke results](docs/smoke_results.md). Each design component is marked built, partial or TODO.

## Installable SDK & CLI (`labrl`)

Install the unified end-to-end SFT-to-RLVR post-training library (`SFT` → `DPO` / `MPO` / `TROPD` → `PPO` / `GRPO` / `GDPO` + `DAPO` → `Eval`) with cryptographic holdout `SplitGuard` and dual execution backends (`TorchLocalEngine` for CPU/1×GPU and `NeMoRLEngine` for 4–8×A100/H100):

```bash
pip install -e .
labrl list
labrl pipeline --corpus calisol23 --methods sft,dpo,mpo,ppo,grpo --smoke
labrl recipe --algo gdpo_multireward --domain calisol_conductivity --model qwen3_8b --hardware node_8xa100_colocated --dump-yaml outputs/calisol_8xa100.yaml
```

```python
from labrl import PostTrainPipeline

manifest = PostTrainPipeline.from_methods(
    ["sft", "dpo", "mpo", "ppo", "grpo"],
    corpus="calisol23",
    smoke=True,
).run()
```

## Post-training

`posttrain/` is a separate stack for SFT, DPO, PPO, and GRPO on the BioProBench checkers. The reference policy for the KL term is the SFT checkpoint, and KL stays on by default. The write-up is [docs/posttrain/README.md](docs/posttrain/README.md). Losses, the reward ladder, hacking checks, contamination, evaluation, and the KL anchor each have a page under `docs/posttrain/`.

```bash
python -m pytest -q
python -m posttrain.sft --smoke
python -m posttrain.dpo --smoke
python -m posttrain.ppo --smoke
python -m posttrain.grpo --smoke
```

Smoke uses a tiny random model on CPU. It is not a lab result. GPU runs are described in `scripts/colab/posttrain_colab.py` and `notebooks/colab_posttrain.ipynb` and are **not run yet**. LAB-Bench stays evaluation only. BioProBench is CC BY-NC 4.0.

## Run locally

Use the existing CPU environment, or create a venv and install `requirements.txt`. The checked-in `data/v0` files are sufficient for tests and training smoke runs; no raw-data download is needed.

```bash
.venv/bin/python -m pytest -q
bash scripts/smoke_all.sh
.venv/bin/python scripts/train_sft.py configs/scientist/sft_smoke.yaml --smoke
.venv/bin/python operator/world_model.py --smoke
```

All training entry points accept `[config.yaml] --smoke` and optional `key=value` overrides for regular runs. Smoke mode forces tiny CPU settings and a locally initialized GPT-2 byte model. It uses real short examples and the actual TRL/PyTorch training paths. It does not download a model. `scripts/smoke_all.sh` runs all training methods in both tracks and writes real timings, losses, return codes and profile summaries to `outputs/smoke_report.json`. Logs and adapters stay in `outputs/smoke/`.

GRPO smoke rewards and losses can be zero because a random model does not emit valid structured answers. This is reported directly. Per-step profiles are under each run's `profile/` directory; losses are in metrics files and plain-loop step logs.

For full GPU jobs, use [Scientist Colab](notebooks/colab_scientist.ipynb) or [Operator Colab](notebooks/colab_operator.ipynb). Unpack the current workspace snapshot including `data/v0` at `/content/labbench-posttrain`; exclude `.git`, `.venv`, `outputs` and `data/raw`. The notebooks enforce an A100, run the full configs, save adapters/metrics/profiles locally, and evaluate the SFT adapter. Methods start as independent baselines; sequential SFT-to-DPO/GRPO needs an explicit checkpoint override. No GPU job has been run here.

## Data from real files

Computed from `data/v0/records.parquet`, `robot_frames.parquet` and task JSONL/stat files:

- 49,139 records from 20 sources.
- Split: 40,790 train / 4,189 validation / 4,160 test; 0 groups cross these checked-in group splits.
- 11,168 records have coarse protocol steps; 10,987 have an explicitly false outcome. Unknown outcomes stay null.
- 33,585 attached annotations: 32,565 automatic and 1,020 source-author human annotations.
- 20,517 robot frames. 23,880 SFT examples and 9,867 preference pairs.

| evidence origin | records |
|---|---:|
| measurement | 29,399 |
| compute | 13,990 |
| ml_prediction | 5,667 |
| human_annotation | 83 |

| source | records |
|---|---:|
| chembl | 10,769 |
| pubchem_compound | 8,687 |
| pubchem_bioassay | 7,545 |
| ord | 6,212 |
| qm9 | 5,000 |
| open_x_embodiment/taco_play | 3,603 |
| ada_sdl_log | 2,134 |
| open_x_embodiment/toto | 1,003 |
| open_x_embodiment/berkeley_autolab_ur5 | 1,000 |
| alab_precursor_genome | 995 |
| platero_cytation3 | 768 |
| open_x_embodiment/nyu_franka_play_dataset | 456 |
| alab_xrd_raw | 400 |
| darus_hydrolase_hplc | 212 |
| open_x_embodiment/droid_100 | 100 |
| eln_ca_imaging | 83 |
| growth_curves_cytation5 | 72 |
| microscopy_alemany_fig2c | 64 |
| chemos2_log | 33 |
| rodare_cmpd10a | 3 |

Measurement, compute, ML prediction and human annotation stay separate. Modality, physical scale, source, units, conditions, licenses and raw pointers are retained. Automatic action parses and synthetic preference corruptions are weak training labels, not lab observations. Real success/failure pairs are restricted to the same split; matching by family still needs stronger experimental controls.

## Splits, assay quality and benchmarks

Rebuild only from existing `data/interim` files. These commands write a separate dataset and leave the checked-in group split in place:

```bash
.venv/bin/python scripts/build_dataset.py --out outputs/time_split --split-mode time --val-year 2009 --test-year 2011 --prospective-source platero_cytation3
.venv/bin/python scripts/make_tasks.py --data-dir outputs/time_split
.venv/bin/python scripts/assay_quality.py
.venv/bin/python scripts/benchmark_link.py
```

Time mode uses `conditions.document_year` where present; undated records fall back to group splitting. Only 4,694 current records have a document year. Time splits can repeat scaffolds across years and are not equivalent to unseen-group tests. The datacard records cutoffs, fallback counts and group overlap. `--prospective-source`, `--prospective-id-file` (one ID per line) and `--prospective-fraction` reserve entire groups in `prospective_blind.parquet`. Routine task and pair generation excludes those answers. This is an offline reservation; real prospective blind lab tests remain TODO.

Assay quality reports sample CV by endpoint plate/sample/concentration. Z′ is unavailable in the current exports because no positive controls are labeled; samples are not relabeled as controls. H/L growth curves have uncertain replicate meaning. Assay-calibrated RM training remains TODO.

The merged catalog is a discovery list. The generated map covers every row using actual catalog columns and local sources/tasks. Related interfaces or a local family subset do not mean a benchmark was run. Access and license metadata still need source-specific review.

## Actions and logs

`src/labpt/actions.py` defines JSON-Schema instrument calls, strict Data/Error results, and run-log T1 completion rewards plus observed recovery pairs. Skips and retries alone do not count as successful execution. Pass planned step IDs to penalize omissions.

`operator/check_constraints.py --plan PLAN.json --constraints CONSTRAINTS.json` returns a reward for trusted workspace bounds, joint limits and swept-point collision boxes. Skills, waypoints and extra avoid-box constraints are model outputs; trusted workspace/joint/obstacle limits belong to the caller. Full IK and arm collision checking are TODO. `src/labpt/logs.py` reads/writes topic/timestamp/message JSONL with Unix-nanosecond timestamps; it is MCAP-style, not binary MCAP.

`data/raw` is left alone. `.venv/`, `outputs/`, raw downloads and interim files are ignored. No HuggingFace uploads are part of this workflow. See [data licenses](DATA_LICENSES.md).

## Protocol reasoning

The scientist and operator tracks are unchanged. This section is the LAB-Bench harness and the BioProBench environments.

| path | what it is |
|---|---|
| `labeval/` | LAB-Bench loader, runner, and summary scripts |
| `bioenvs/` | PQA, ORD, ERR, and GEN environments with code-checked rewards |
| `common/llm.py` | OpenRouter client. Reads `OPENROUTER_API_KEY` from the environment or a local `.env` (gitignored). Responses cache under `results/.cache` |
| `train/grpo_example.py` | Example of passing `env.reward_fn` to TRL GRPO. Not run |
| `data/labbench/` | Public LAB-Bench text. CC-BY-SA-4.0. Do not train on it |
| `data/bioprobench/` | BioProBench test splits and a 500-row train sample per task. CC BY-NC 4.0. Not MIT |
| `results/protocol_reasoning/` | Summaries from the completed run. Raw per-question model outputs are not included |
| `tests/test_envs.py` | Reward checks. No API key |
| `docs/protocol_reasoning.md` | Design write-up and the reported numbers |

The completed run compared Haiku 5.5, GPT 6 Luna, and GLM 5.3 flash on public LAB-Bench. The write-up and the files under `results/protocol_reasoning/` are those results. This repo does not recompute them.

LAB-Bench data is CC-BY-SA-4.0 ([upstream repo](https://github.com/Future-House/LAB-Bench)). `data/labbench/LICENSE-LAB-Bench` is that license, and the jsonl files keep the canary string. Do not train on `data/labbench`. FigQA and TableQA images are not in this repo. Clone LAB-Bench and set `LABBENCH_REPO` to that clone before scoring those two families.

BioProBench data is CC BY-NC 4.0 ([upstream repo](https://github.com/YuyangSunshine/bioprobench), also linked as bioprotocolbench). Do not treat it as MIT. `bash scripts/get_bioprobench.sh` downloads the full train splits and replaces the 500-row samples.

```bash
python -m pytest tests/test_envs.py -q          # reward checks, no API key
python -m bioenvs.audit                         # scripted exploit policies, no API key
python -m labeval.run --seeds 0 1               # needs OPENROUTER_API_KEY; full run is paid
python -m labeval.run --limit 2                 # small paid smoke
python -m bioenvs.baseline --n 100 --tag greedy # needs OPENROUTER_API_KEY
```

`python -m labeval.analyze` reads per-question jsonl files from a run. Those raw outputs are not shipped, so analyze cannot rebuild the summaries from this checkout. Use the files already in `results/protocol_reasoning/`. The same is true for `python -m labeval.budget` and `python -m labeval.counterfactual`: the summary files are included, the raw model text is not.

Use an environment from your own trainer:

```python
from bioenvs import make_env
env = make_env("err", split="train", seed=0)
obs = env.reset()
reward, done, info = env.step(model_output)
rows = env.dataset(1000)          # [{prompt, info, id}]
# GRPOTrainer(reward_funcs=env.reward_fn, ...)  see train/grpo_example.py
```
