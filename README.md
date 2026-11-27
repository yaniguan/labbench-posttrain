# labbench-posttrain

A small lab post-training project with two tracks, plus a protocol-reasoning eval and four reward environments:

- **Scientist:** learn scientific representations, align modalities with text, answer historical questions with evidence labels, and run SFT, DPO, reward modeling, GRPO and distillation.
- **Operator:** turn protocols into actions, learn state/action behavior and a world model, and score structured plans against simple physical constraints.
- **Protocol reasoning:** score public LAB-Bench questions, and train against four BioProBench tasks with code-checked rewards. See [Protocol reasoning](#protocol-reasoning) and [the write-up](docs/protocol_reasoning.md).

> **Auto-Lab Ecosystem**: Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput` in [auto-lab](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)

**GPU results: PENDING (A100 Colab).** CPU smoke checks verify code execution. They do not establish useful scientific accuracy or safe robot control.

See [the design mapped to code](docs/DESIGN.md), [architecture and history](docs/ARCHITECTURE_AND_HISTORY.md), [dataset overview](docs/dataset_overview.md), [benchmark map](docs/benchmark_map.md), [assay quality](docs/assay_quality.md), and [CPU smoke results](docs/smoke_results.md). Each design component is marked built, partial or TODO.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full architectural documentation, dataset schemas, mathematical objectives, commit lineage, and sibling-repository contracts are maintained in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

### Object
End-to-end **SFT-to-RLVR (Reinforcement Learning with Verifiable Rewards)** post-training, benchmark evaluation, and anti-hack auditing stack (`L3`) for wet-lab science (**Scientist** track), physical robot/liquid-handler execution (**Operator** track), biological protocol troubleshooting (**LAB-Bench** / **BioProBench**), and experimental electrolyte design (**CALiSol-23**), packaged as the installable `labrl` (`v0.2.0`) SDK and CLI.

### Data
- **Lab v0 (`data/v0/`)**: `49,139` provenance-separated records (`29,399` measurement, `13,990` compute, `5,667` ML prediction, `83` human annotation) from `20` sources (`40,491` train / `4,321` val / `4,327` test, `0` `cid|` fallback keys, `0` empty `scaffold|` keys, `0` cross-split canonical scaffold/tautomer/counterion groups), `20,517` robot frames, `23,880` SFT examples, and `9,867` preference pairs.
- **CALiSol-23 (`data/calisol/`)**: `13,825` raw $\rightarrow$ `13,228` clean experimental Li-ion electrolyte conductivity measurements across `27` publications (`9,526` train / `1,804` val / `1,898` test with zero DOI overlap; `CC-BY-4.0`).
- **BioProBench (`data/bioprobench/`)**: `2,000` train sample rows (`500` $\times$ `4` tasks: `PQA`, `ORD`, `ERR`, `GEN`) and `4,333` test rows (`CC BY-NC 4.0`).
- **LAB-Bench (`data/labbench/`) & Catalog (`benchmarks/`)**: `1,967` evaluation-only questions across `31` JSONL files (`CC-BY-SA-4.0`, canary-guarded) plus a `205`-entry benchmark discovery catalog (`benchmarks/catalog.csv`).
- **2024–2026 Open Science Registry (`src/labrl/data/open_datasets_2026.py`, `scripts/ingest_open_science_2026.py`)**: `9` live-verified Hugging Face / GitHub datasets fetched on-demand into gitignored `data/open_science_2026/`—`5` trainable corpora with `SplitGuard` canonical Murcko/acyclic scaffold & DOI union-find isolation (`0` cross-split leakage: `roman-bushuiev/MassSpecGym` `MIT`, `osunlp/SMolInstruct` `CC-BY-4.0`, `zjunlp/Mol-Instructions` `CC-BY-4.0`, `maomlab/AqSolDB` `CC-BY-4.0`, `open-reaction-database/ord-data` `CC-BY-SA-4.0`) and `4` strictly `eval_only=True` canary-protected benchmarks (`jablonkagroup/ChemBench` `MIT`, `m-a-p/SuperGPQA` `ODC-By-1.0`, `futurehouse/ether0-benchmark` `CC-BY-4.0`, `futurehouse/BixBench` `Apache-2.0`).

### Model
- **Policy Backbones**: Tiny offline byte-level GPT-2 models (`build_tiny_lm`, `ensure_tiny_model`) for zero-network CPU smoke verification; target configs for `Qwen3-1.7B/8B`, `Qwen2.5-7B`, `Qwen2.5-VL-7B`, `Llama-3.1-8B`, and `OLMo-2-7B` with architecture-aware PEFT LoRA and causal PPO value heads.
- **Multimodal & Embodied Modules**: `MolEncoder` (Morgan FP MLP), `SpectrumEncoder` (1D CNN), `ImageEncoder` (2D CNN), `TextEncoder`, soft-token LLM projector (`scientist/`), MLP behavior cloning (`operator/train_bc.py`), and state-delta world model (`operator/world_model.py`).
- **Symbolic Verifiers**: `bioenvs/` (`PQA`, `ORD`, `ERR`, `GEN`, `CALiSolDesign`, `CALiSolPredict`), 8 anti-hack `VerifiedRewardTemplate` classes (`src/labrl/rewards/`), Pydantic `InstrumentAction` schemas, and 3D swept-segment obstacle checkers (`src/labpt/`).

### Training
- **Algorithms**: Completion-masked `SFT` $\rightarrow$ offline `DPO` / `ORPO` / hybrid `MPO` (`DPO + BCO + SFT` with EMA `reward_shift`) / `KD` & online multi-teacher `TROPD` $\rightarrow$ online `PPO` (`VAPO` length-adaptive $\lambda$ + positive NLL) and `GRPO` (`GDPO` per-channel standardization, `DAPO` clip-higher & dynamic resampling, `GSPO` sequence ratios, `CISPO`, `TIS`/`ICE-POP`).
- **Execution Status**: `15/15` Scientist/Operator CPU smoke jobs (`63.96s`) and `posttrain`/`labrl` unit & smoke suites pass locally; `results/protocol_reasoning/` stores completed OpenRouter LAB-Bench summaries (`Haiku 5.5`, `GPT 6 Luna`, `GLM 5.3 flash`). **All single-GPU A100 Colab and multi-GPU NeMo-RL training runs are PENDING.**

### History
Evolved from initial `v0` scaffolding and the 205-entry benchmark catalog (`63697fb`, `c247afe`, `b2bde46`, Nov 3–4, 2026) through the LAB-Bench/BioProBench harness (`101933a`) and `posttrain/` stack (`dbec763`, Nov 10), Milestones 1–3 (`34c1ccf` Reverse-KL & VAPO/GSPO/ICE-POP, `cdc897c` v2 anti-hack verifiers & GDPO/DAPO/NeMo-RL recipes, `18b89b3` MPO & TROPD, Nov 11), CALiSol-23 RLVR suite (`0a306cd`), installable `labrl` SDK/CLI (`6cb46a7`), and 8 verified reward templates (`94f0cf5`, Nov 12, 2026).

### Unique Properties
- **Score-Function Reverse-KL Anchor (Per-Token & Sequence-Exact)**: Restores unbiased per-token $\nabla_\theta D_{\text{KL}}(\pi_\theta(y_t \mid x, y_{<t}) \parallel \pi_{\text{SFT}})$ gradients on Schulman's `k3` estimator (matching `yaniguan/RL` `reference_policy_kl_type: k3`), and provides an exact multi-token sequence-level estimator (`sequence_reverse_kl_reward_to_go` / `kl_mode="sequence_exact"`) carrying causal future KL suffix sums while enforcing frozen SFT anchoring by default.
- **Anti-Hack Scientific RLVR**: Combines conservative experimental-atlas kernel support gating, multi-channel GDPO normalization, dimensional unit conversion, SMILES stoichiometry checks, spurious-reward control ladders, and cryptographic `SplitGuard` + LAB-Bench canary enforcement.
- **Ecosystem Interfaces**: Bridges `RL` (NeMo-RL cluster backend), `YLM`, `WignerFlow`, `assayfold`, `twinstep`, `labfill` (`L0`/`L3`) and post-execution causal blame (`lab-attrib`, `L4`) with pre-execution digital-twin gates (`lab-rehearsal`, `lab-verify`), robot/edge drivers (`stepbot`, `lab-hands`, `lab-feet`), and epistemic planners (`lab-decide`, `lab-ask-why`).

## Repository Layout

```text
labbench-posttrain/
├── .github/workflows/ci.yml       # CPU pytest + 15-job smoke CI workflow
├── benchmarks/                    # 205-entry scientific & robotic benchmark catalog + charts
├── bioenvs/                       # Verifiable RL envs (PQA, ORD, ERR, GEN, CALiSol-23) + exploit audit
├── common/llm.py                  # Async OpenRouter client with disk cache & cost tracking
├── community_extensions/          # Twin-delta step-level process credit & anti-<think>-bloat shaping
├── configs/                       # Scientist, Operator, posttrain, KL-grid, NeMo-RL & 4-axis recipes
├── data/                          # v0 unified parquets/tasks, BioProBench, CALiSol-23, LAB-Bench (eval)
├── docs/                          # Architecture/history, design map, posttrain math & protocol write-ups
├── labeval/                       # Public LAB-Bench loader, runner, token-budget & counterfactual scripts
├── lean/                          # Lean 4 formalization (`LabPostTrainLean.lean`) of SplitGuard & reward gates
├── notebooks/                     # Pending A100 Colab notebooks (Scientist, Operator, Post-training)
├── operator/                      # Behavior cloning, state-delta world model & 3D constraint checker
├── posttrain/                     # Standalone PyTorch SFT, DPO, MPO, TROPD, PPO, GRPO, KL & anti-hack suite
├── results/                       # Protocol-reasoning LAB-Bench & BioProBench baseline summaries
├── scientist/                     # Multimodal encoders, InfoNCE alignment, soft-token projector & KD
├── scripts/                       # Dataset ingesters, split builders, TRL trainers, smoke & CALiSol runners
├── src/
│   ├── labpt/                     # Record schema, provenance splits, typed InstrumentActions & rewards
│   ├── labpt_nemo_rl/             # Loose-coupled NeMo-RL EnvironmentInterface shim & recipe validator
│   └── labrl/                     # Unified v0.2.0 SFT-to-RLVR SDK, CLI & 8 VerifiedRewardTemplates
├── tests/                         # Unit, math-gradient, anti-hack, split-leakage, wire & SDK/CLI test suites
├── autolab_wire.py                # 15-repo Auto-Lab wire adapter (`autolab-wire/v1`, Layer L3)
├── DATA_LICENSES.md               # Source-by-source dataset licensing & canary constraints
├── LICENSE                        # MIT License (code)
├── pyproject.toml                 # Installable `labrl` package & CLI entry point
└── requirements.txt               # Full Python training & evaluation dependencies
```

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "labbench-posttrain"`, `LAYER = "L3"`) implements `process_packet(packet: dict) -> dict` as the post-training and memory-admission node closing the 15-repository Auto-Lab loop (`lab-ears` -> `lab-ask-why` -> `lab-decide` -> `lab-automation-atlas` -> `buy-or-make` -> `labshift` -> `lab-rehearsal` -> `lab-verify` -> `stepbot` -> `lab-hands` -> `lab-feet` -> `lab-attrib` -> `twinstep` -> `labfill` -> `labbench-posttrain`):

- **Upstream Wire Inputs (`lab-attrib`, `twinstep`, `labfill` & Halted Gates)**:
  - Consumes post-execution causal attribution (`attribution.admit_to_memory` / `L4_ATTRIB_QUARANTINE`) from [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`), seeded protocol error-twin DAGs (`error_twins`) from [`twinstep`](https://github.com/yaniguan/twinstep) (`L3`), and packed real-token counts (`packed_batch.total_real_tokens`) from [`labfill`](https://github.com/yaniguan/labfill) (`L3`).
  - Unlike upstream physical actuators, `labbench-posttrain` **also scores halted packets** so RLVR learns calibrated abstentions (`L6_ABSTAIN` -> `gated_reward = 0.25`, `admitted_to_sft_rlvr = False`) and zero-reward safety/quarantine rejections (`L4_ATTRIB_QUARANTINE`, `L4_REHEARSAL_BLOCK`, `L1_MCP_BLOCK*` -> `gated_reward = 0.0`, `admitted_to_sft_rlvr = False`) without poisoning SFT or the certified ledger.
- **Formal `SplitGuard` & Multiplicative Anti-Hack Gate (`LabPostTrainLean`)**:
  - Enforces obfuscation-hardened `SplitGuard` holdout disjointness and recursive `LAB-Bench` canary isolation across all string fields under NFKC, zero-width/control stripping, Cyrillic/Greek homoglyphs, and Base64/hex/percent/escape decoding (`L3_SPLITGUARD_CANARY_BLOCK`, `gated_reward = 0.0`).
  - Enforces `unverified_rollout_gets_zero_reward` in `autolab_wire.process_packet`, while `posttrain.grpo.train_grpo` invokes `check_kl_anchor` (`kl_anchor_drift_bound`) on every step and `check_gdpo_pareto_weights` on multi-channel `GDPO` batches (`LabPostTrainLean` proves both raw-channel simplex bounds `raw_channel_pareto_normalized_le_max` and standardized within-group GDPO zero-sum conservation `gdpo_standardized_multi_channel_sum_zero`).
  - Fails closed (`L3_WIRE_SCHEMA_REJECT`, `halted = True`, `gated_reward = 0.0`, zero memory admission) on unknown attribution outcomes (`"FAILED"`, `"UNKNOWN"`, `"CERTIFIED NEGATIVE"`), non-boolean strings (`"false"`), malformed token counts (`None`, `"many"`), or any `admit_to_memory = True` packet lacking a valid 64-hex `ledger_entry_sha256` digest; separates `CERTIFIED_POSITIVE` (`gated_reward = 1.0`, `admitted_to_sft = True`, `dpo_role = "chosen"`, `mpo_bco_label = 1`) from `CERTIFIED_NEGATIVE` physical disproofs (`gated_reward = 0.0` on synthesis maximization, `admitted_to_sft = False`, `admitted_to_preference_memory = True`, `preference_role = "certified_negative_truth"`, `dpo_role = "rejected"`, `mpo_bco_label = 0`) and `L4_ATTRIB_QUARANTINE` characterization artifacts (`admitted_to_preference_memory = False`, `dpo_role = None`).
- **Downstream Closed-Loop Handoff (`lab-decide` & `lab-feet`)**:
  - Emits `posttrain_update` and builds mixed DPO/BCO preference corpora via `build_wire_mpo_dataset`, feeding calibrated epistemic policies back into [`lab-decide`](https://github.com/yaniguan/lab-decide) (`L6` split-conformal trust-region gate) and distilled policy checkpoints into [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L0` INT8 / 4-bit bench-side edge runtime).

## Installable SDK & CLI (`labrl`)

Install the unified end-to-end SFT-to-RLVR post-training library (`SFT` → `DPO` / `MPO` / `TROPD` → `PPO` / `GRPO` / `GDPO` + `DAPO` → `Eval`) with cryptographic holdout `SplitGuard` and dual execution backends (`TorchLocalEngine` for CPU/1×GPU and `NeMoRLEngine` for 4–8×A100/H100):

```bash
pip install -e .
pip install -e ".[nemo]"   # pins yaniguan/RL @ a7299bc888e4b001020493d976f722513e57e68c for Ray GRPO/GSPO
labrl list
labrl pipeline --corpus calisol23 --methods sft,dpo,mpo,ppo,grpo --smoke
labrl pipeline --corpus calisol23 --methods sft,mpo,gdpo --sequential --smoke
labrl recipe --algo gdpo_multireward --domain calisol_conductivity --model qwen3_8b --hardware node_8xa100_colocated --dump-yaml outputs/calisol_8xa100.yaml

# Register BioProBench/CALiSol envs and dry-run or launch 1-GPU GSPO on yaniguan/RL (GPU results: PENDING):
python scripts/posttrain/run_nemo_rl.py --algo gspo_longcot --domain bioenvs_pqa_ord_err --model qwen3_1.7b --hardware local_1gpu_colocated --dry-run
python scripts/posttrain/run_nemo_rl.py --algo gspo_longcot --domain bioenvs_pqa_ord_err --model qwen3_1.7b --hardware local_1gpu_colocated --set grpo.max_num_steps=1
```

```python
from labrl import PostTrainPipeline

manifest = PostTrainPipeline.from_methods(
    ["sft", "dpo", "mpo", "ppo", "grpo"],
    corpus="calisol23",
    smoke=True,
).run()
```

### 2024–2026 Open Science Registry & On-Demand Ingestion (`open_datasets_2026`)

`src/labrl/data/open_datasets_2026.py` and `scripts/ingest_open_science_2026.py` register `9` live-verified Hugging Face / GitHub datasets, downloading raw/normalized splits exclusively into gitignored `data/open_science_2026/`:

| Dataset ID | Source (`hf_repo_id`) | SPDX License | Mode | `SplitGuard` / Canary Policy |
|---|---|---|---|---|
| `massspecgym` | [`roman-bushuiev/MassSpecGym`](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) | `MIT` | Trainable (`spectrum`) | `canonical_scaffold_group_key(smiles)` (`0` scaffold leakage) |
| `smolinstruct` | [`osunlp/SMolInstruct`](https://huggingface.co/datasets/osunlp/SMolInstruct) | `CC-BY-4.0` | Trainable (`reaction` / `molecular_property`) | `canonical_scaffold_group_key` (`0` scaffold leakage) |
| `mol_instructions` | [`zjunlp/Mol-Instructions`](https://huggingface.co/datasets/zjunlp/Mol-Instructions) | `CC-BY-4.0` | Trainable (`molecular_property` / `bioassay`) | Canonical scaffold + sequence family digest (`0` leakage) |
| `aqsoldb` | [`maomlab/AqSolDB`](https://huggingface.co/datasets/maomlab/AqSolDB) (`10.7910/DVN/OVHAW8`) | `CC-BY-4.0` | Trainable (`molecular_property`, `logS`) | `canonical_scaffold_group_key(SMILES)` (`0` scaffold leakage) |
| `ord_2026` | [`open-reaction-database/ord-data`](https://huggingface.co/datasets/open-reaction-database/ord-data) | `CC-BY-SA-4.0` | Trainable (`reaction`, `%` yield) | Union-Find `(doi, product_scaffold)` (`0` DOI & `0` scaffold leakage) |
| `chembench` | [`jablonkagroup/ChemBench`](https://huggingface.co/datasets/jablonkagroup/ChemBench) | `MIT` | **`eval_only=True`** | Blocked from training (`SplitLeakageError`) + `26b5c67b-...` canary |
| `supergpqa` | [`m-a-p/SuperGPQA`](https://huggingface.co/datasets/m-a-p/SuperGPQA) | `ODC-By-1.0` | **`eval_only=True`** | Blocked from training (`SplitLeakageError`) + STEM filter & canary |
| `ether0_benchmark` | [`futurehouse/ether0-benchmark`](https://huggingface.co/datasets/futurehouse/ether0-benchmark) | `CC-BY-4.0` | **`eval_only=True`** | Blocked from training (`SplitLeakageError`) + canary stamp |
| `bixbench` | [`futurehouse/BixBench`](https://huggingface.co/datasets/futurehouse/BixBench) | `Apache-2.0` | **`eval_only=True`** | Blocked from training (`SplitLeakageError`) + `bixbench:5c36:` canary |

```bash
python scripts/ingest_open_science_2026.py --list
python scripts/ingest_open_science_2026.py --datasets aqsoldb,massspecgym --limit 200
python scripts/ingest_open_science_2026.py --datasets chembench,supergpqa --allow-eval-download --limit 50
labrl pipeline --corpus open_science_2026 --methods sft,dpo,mpo,grpo --smoke
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
- Split: 40,491 train / 4,321 validation / 4,327 test; 0 `cid|` fallback keys, 0 empty `scaffold|` keys, and 0 groups crossing these checked-in group splits.
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
