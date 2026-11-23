# Architecture and History — `labbench-posttrain`

> **Auto-Lab Ecosystem Role**: Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Training foundation models for autonomous wet-lab science and robotic execution faces three compounding failure modes that generic instruction tuning cannot resolve:
1. **Evidence conflation**: Density-functional theory (DFT) calculations, fitted machine-learning predictions, human notebook annotations, and physical instrument measurements are routinely merged into undifferentiated targets, teaching models to treat noisy heuristics or synthetic priors as ground-truth observations.
2. **Reward hacking on ungrounded proxies**: Reinforcement learning against weak format or bag-of-words verifiers induces degenerate policies—such as repeating a single protocol step, hedging multiple `[ANSWER_START]` spans, farming partial credit via zero-confidence wrong guesses, or extrapolating inverse electrolyte formulations outside experimental support.
3. **Policy drift and pretraining priors**: Without a frozen supervised fine-tuning (SFT) reference anchor, spurious-reward controls (random, format-only, incorrect-label ladders), and contamination/canary gates, rising RL reward curves on pre-cutoff benchmarks cannot distinguish genuine laboratory reasoning from prior sharpening or memorization.

`labbench-posttrain` (packaged as the installable `labrl` Python SDK and CLI) provides an end-to-end **SFT-to-RLVR (Reinforcement Learning with Verifiable Rewards) post-training, evaluation, and anti-hack auditing stack** spanning multimodal scientific reasoning (**Scientist** track), physical robot/liquid-handler execution (**Operator** track), biological protocol troubleshooting (**LAB-Bench** / **BioProBench**), and experimental electrolyte design (**CALiSol-23**).

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `labbench-posttrain` serves as the core post-training and verifiable-reward hub in **`L3 · Post-Training RLVR, Error Twins & GPU Throughput`**:
- **Upstream (`L0`–`L2` Foundation & Perception)**: Consumes base checkpoints and domain representations aligned with [`YLM`](https://github.com/yaniguan/YLM) (pretrained lab language models & `LabLean` formal specs), [`WignerFlow`](https://github.com/yaniguan/WignerFlow) / [`equivariant-gpu-lab`](https://github.com/yaniguan/equivariant-gpu-lab) (3D molecular/equivariant physics), [`assayfold`](https://github.com/yaniguan/assayfold) (biomolecular structure/assay priors), [`ouroboros-ocsr`](https://github.com/yaniguan/ouroboros-ocsr) (chemical structure recognition), and [`lab-ears`](https://github.com/yaniguan/lab-ears) (instrument log/audio/telemetry streams).
- **Peer (`L3` Execution & Error Twins)**: Shares loss/advantage semantics (`ClippedPGLossFn`, `GDPOAdvantageEstimator`, `MPOLossFn`, `OPDAdvantageEstimator`, `ICE-POP`) with [`RL`](https://github.com/yaniguan/RL) (the NeMo-RL multi-node cluster engine) and protocol-error dynamics with [`twinstep`](https://github.com/yaniguan/twinstep) and [`labfill`](https://github.com/yaniguan/labfill).
- **Downstream (`L4`–`L6` Verification, Execution & Metacognition)**: Supplies calibrated action/observable policies and structured JSON/schema verifiers to [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) and [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4` default-deny digital twins), [`stepbot`](https://github.com/yaniguan/stepbot), [`lab-hands`](https://github.com/yaniguan/lab-hands), and [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L1`/`L2`/`L5` robot execution), while feeding calibrated confidence/uncertainty outputs (`CalibratedMCQTemplate`, `ScalarObservableTemplate`, `CALiSolPredict`) into [`lab-decide`](https://github.com/yaniguan/lab-decide) and [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (`L6` epistemic decision and debate brain).

### Inputs, Outputs, and Scope Boundaries
- **Inputs**:
  - Provenance-tagged scientific/robotic records (`data/v0/records.parquet`, `robot_frames.parquet`, `tasks/*.jsonl`), BioProBench tasks (`data/bioprobench/*.json`), CALiSol-23 conductivity measurements (`data/calisol/calisol23.csv`), public LAB-Bench evaluation items (`data/labbench/*.jsonl`), and 4-axis recipe YAML configs (`configs/recipes/{algo,domain,model,hardware}/`).
- **Outputs**:
  - Trained/smoke-verified model checkpoints and PEFT LoRA adapters (`outputs/`), lineage manifests (`PipelineManifest` / `lineage_manifest.json`) with cryptographic `SplitGuard` signatures, NeMo-RL cluster launch YAMLs, `RewardPacket` scalar + GDPO channel tensors, LAB-Bench evaluation summaries (`results/protocol_reasoning/`), and reward-hacking / contamination audit reports.
- **Scope Boundaries**:
  - **All neural training runs in this repository to date are CPU smoke executions on tiny in-memory random models (`build_tiny_lm` / `ensure_tiny_model`).** Full GPU training on Qwen2.5/Qwen3/Llama/OLMo checkpoints (`1x`–`8x` A100/H100 via Colab or NeMo-RL) is **PENDING** and never claimed as completed.
  - Operator constraint checks (`src/labpt/constraints.py`) perform swept-point axis-aligned bounding-box and joint-range checks, not full URDF/MJCF link collision or inverse kinematics (IK) certification.
  - LAB-Bench (`data/labbench/`) is strictly evaluation-only (`CC-BY-SA-4.0`) and guarded by canary checks.

---

## 2. Data

### Summary of Checked-In Datasets & Catalogs

| Dataset / Asset | Location | Volume | Splits & Grouping | Provenance / Modality | License |
|---|---|---|---|---|---|
| **Unified Lab v0 Corpus** | `data/v0/records.parquet`, `{train,val,test}.parquet` | `49,139` records from `20` sources | `40,790` train / `4,189` val / `4,160` test (`0` cross-split groups via `Record.group_key`) | `29,399` measurement, `13,990` compute, `5,667` ML prediction, `83` human annotation | Per-record (`CC0-1.0`, `MIT`, `CC-BY-4.0`, `CC-BY-SA-3.0/4.0`, NCBI PD; see `DATA_LICENSES.md`) |
| **Robot State/Action Frames** | `data/v0/robot_frames.parquet` | `20,517` frames | Leak-free episode group splits matching `records.parquet` | Open X-Embodiment (`taco_play`, `toto`, `berkeley_autolab_ur5`, `nyu_franka_play`, `droid_100`) | `CC-BY-4.0` (upstream OXE/DROID) |
| **Lab v0 Task Views** | `data/v0/tasks/{sft,pairs}_{train,val,test}.jsonl` | `23,880` SFT rows, `9,867` preference pairs | Derived from `train`/`val`/`test` excluding `prospective_blind` | `protocol2actions`, `provenance`, `outcome`, `safety_failure`, `property_qa`; paired `wrong_provenance`, `corrupted_actions`, `success_vs_failure` | Inherits source record licenses |
| **CALiSol-23 Conductivity Atlas** | `data/calisol/calisol23.csv`, `calisol_smiles.csv` | `13,825` raw rows -> `13,228` clean measurements (`129` isothermal design regimes) | Publication-level (`doi`) zero-overlap split: `9,526` train (`15` DOIs) / `1,804` val (`5` DOIs) / `1,898` test (`7` DOIs); `85`/`22`/`22` design tasks | Experimental liquid Li-ion electrolyte ionic conductivity $k$ ($\text{mS/cm}$) across `13` salt labels & `38` organic solvents | `CC-BY-4.0` (de Blasio et al., *Sci. Data* 2024; DTU Data) |
| **BioProBench RLVR Tasks** | `data/bioprobench/{PQA,ORD,ERR,GEN}_{train,test}.json` | `2,000` train sample (`500` $\times$ `4`) + `4,333` test (`1,200` PQA, `1,161` ORD, `1,200` ERR, `772` GEN) | Official `train`/`test` (`scripts/get_bioprobench.sh` fetches full `47k`/`28k`/`93k`/`107k` train splits) | Biological protocol QA, step ordering, fault detection/repair, protocol generation | `CC BY-NC 4.0` (**non-commercial**, not MIT) |
| **Public LAB-Bench (Eval Only)** | `data/labbench/*.jsonl` (`31` files) | `1,967` questions across `8` families (`741` core protocol + `1,226` lit/db/fig/table) | Evaluation-only (**never** used for training); carries canary `lab-bench:5c41:26b5c67b-...` | `ProtocolQA` (`108`), `CloningScenarios` (`33`), `SeqQA` (`600`), `LitQA2`, `DbQA`, `SuppQA`, `FigQA`, `TableQA` | `CC-BY-SA-4.0` (FutureHouse) |
| **Benchmark Discovery Catalog** | `benchmarks/catalog.{csv,json,parquet}` | `205` benchmarks | Mapped 1-to-1 to local tracks/interfaces in `docs/benchmark_map.md` | Scientist (`text/protocol`, `spectra`, `3D`, `tabular`) & Operator (`robot action/video`, `instrument log`) | Metadata catalog (each benchmark retains upstream license) |

### Contamination, Holdout, and Provenance Guards
1. **Explicit Provenance Schema (`src/labpt/schema.py::Record`)**: Separates `measurement`, `compute`, `ml_prediction`, and `human_annotation` alongside `Modality` (9 classes) and `Scale` (6 classes). Weak rule-parsed actions (`annotator="rule:v0"`, `auto=True`) and synthetic preference corruptions are never relabeled as experimental readouts.
2. **Group, Chronological, and Prospective-Blind Splitting (`src/labpt/splits.py`, `src/labrl/data/__init__.py::SplitGuard`)**:
   - `assign_splits` supports deterministic MD5 group hashing (`80/10/10`), chronological `conditions.document_year` splits, and whole-group offline `prospective_blind` reservation.
   - `SplitGuard` computes a 16-hex-character SHA-256 signature over `(group_key, val_groups, test_groups)`, enforces disjointness (`assert_disjoint`), and scans training prompts/completions for LAB-Bench canary strings (`CANARY_NEEDLES`).
3. **Zero-DOI-Leakage Electrolyte Splits (`bioenvs/calisol.py`)**: Partitioning CALiSol-23 strictly by publication `doi` prevents VTF temperature-sweep interpolation leakage across `train`, `val`, and `test`.

---

## 3. Model

`labbench-posttrain` spans three model tiers across its packages:

### 1. Causal LM & VLM Policy Backbones (`posttrain/`, `src/labrl/`, `scripts/`)
- **Offline CPU Smoke Backbone (`posttrain/tiny.py::build_tiny_lm`, `src/labpt/tiny.py::ensure_tiny_model`)**: In-memory / local 2-layer or 1-layer byte-level `GPT2LMHeadModel` (`hidden_size = 32` or `64`, `vocab_size = 260`–`300`) with zero network downloads, exercising identical PyTorch/TRL/PEFT code paths in CI.
- **Target Production Backbones (`configs/recipes/model/`, `configs/posttrain/`)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, and `allenai/OLMo-2-1124-7B-Instruct`.
- **Adapters & Heads**:
  - Architecture-aware PEFT LoRA (`posttrain/batching.py::maybe_lora`) targeting `c_attn` (`fan_in_fan_out=True`) on GPT-2 and `q/k/v/o/gate/up/down_proj` on Qwen/Llama/OLMo.
  - PPO scalar critic head (`nn.Linear(hidden_size, 1)`) causally aligned to `hidden_states[-1][:, :-1, :]` so state $s_t$ predicts token $x_{t+1}$.
  - Sequence-classification Bradley-Terry reward model (`AutoModelForSequenceClassification(num_labels=1)` + LoRA in `scripts/train_rm.py`).

### 2. Scientist Multimodal Encoders & Soft-Token Projector (`scientist/`)
- **`MolEncoder`**: 2048-bit radius-2 RDKit Morgan fingerprint $\rightarrow$ 2-layer MLP (`2048 -> 512 -> d`) trained on QM9 (`gap`, `homo`, `lumo`, `mu`, `u0`).
- **`SpectrumEncoder`**: Resampled 256-point z-scored 1D instrument curve (XRD, HPLC, IR, growth curves) $\rightarrow$ 3-layer strided `Conv1d` + `AdaptiveAvgPool1d(8)` + linear projection.
- **`ImageEncoder`**: $32 \times 32$ microscopy patch $\rightarrow$ 2-stage `Conv2d` + `MaxPool2d` denoising encoder.
- **`TextEncoder`**: 4096-bin CRC32 unigram/bigram `log1p` bag-of-ngrams $\rightarrow$ MLP (`scientist/train_align.py` symmetric InfoNCE alignment).
- **Multimodal Soft-Token Projector (`scientist/train_projector.py`)**: 2-layer GELU MLP mapping frozen `MolEncoder` embeddings into $K$ continuous soft prefix tokens prepended to causal LM input embeddings (with SMILES omitted from text prompts).

### 3. Operator Policies, World Model & Symbolic Verifiers (`operator/`, `src/labpt/`, `bioenvs/`, `src/labrl/rewards/`)
- **Behavior Cloning (`operator/train_bc.py`)**: Per-robot state-normalized 3-layer ReLU MLP (`SD -> hidden -> hidden -> AD`) benchmarked against a predict-the-train-mean baseline.
- **Residual State-Delta World Model (`operator/world_model.py::WM`)**: Predicts normalized state increments $\hat{s}_{t+1} = s_t + \sigma_s \cdot f_\theta((s_t - \mu_s)/\sigma_s, a_t)$, evaluated on $k$-step open-loop rollouts against a copy-state baseline.
- **Symbolic & Statistical Verifiers (Zero Neural Judge)**:
  - **BioProBench (`bioenvs/{pqa,ordering,err,gen}.py`)**: Brier-calibrated MCQ (`PQA`), permutation bijection + Kendall's $\tau$ (`ORD`), SequenceMatcher normalized edit-improvement fault repair (`ERR`), and cluster-deduplicated Hungarian step F1 + Kendall's $\tau$ + regex quantity recall (`GEN`).
  - **CALiSol-23 (`bioenvs/calisol.py`)**: Isothermal Nadaraya-Watson top-$k$ kernel conductivity estimator with exponential OOD support gate $\exp(-d_{\min}/\tau)$ (`calisol_design`) and log-space accuracy + Gaussian uncertainty score (`calisol_predict`).
  - **8 Canonical `VerifiedRewardTemplate` Classes (`src/labrl/rewards/templates.py`)**: `calibrated_mcq`, `scalar_observable` (with `convert_scientific_unit`), `inverse_design`, `step_ordering`, `fault_repair`, `action_plan`, `molecular_design` (pure-Python/RDKit `parse_smiles_invariants`), and `spectral_peaks`, composable via `GatedRewardTemplate` and `CounterfactualPairScorer`.
  - **Operator Safety & Schema Checkers (`src/labpt/{actions,constraints}.py`)**: Discriminated Pydantic `InstrumentAction` (`aspirate`, `dispense`, `move`, `heat`, `wait`, `measure`), `ActionResult` Data/Error mutual exclusion, and 3D slab-intersection `segment_hits_box` trajectory verification.

---

## 4. Training

### Post-Training Lifecycle (`SFT` $\rightarrow$ `DPO` / `MPO` / `TROPD` $\rightarrow$ `PPO` / `GRPO` / `GDPO`)

| Stage / Algorithm | Implementation | Objective & Key Mathematical Primitives | Default KL / Anchor Policy |
|---|---|---|---|
| **Stage 1: Cold-Start SFT** | `posttrain/sft.py`, `scripts/train_sft.py` | Completion-masked causal NLL (`sft_nll`), supporting token-level (`DAPO`) or sequence-level normalization; freezes output into `sft_reference` and `sft_anchor`. | Establishes the frozen $\pi_{\text{SFT}}$ reference anchor |
| **Offline DPO / ORPO** | `posttrain/dpo.py`, `scripts/train_pref.py` | $-\log\sigma(\beta[(\log\pi(y_w)-\log\pi_{\text{ref}}(y_w)) - (\log\pi(y_l)-\log\pi_{\text{ref}}(y_l))])$ + optional chosen-NLL anchor (`sft_coef`) + `likelihood_displacement` tracking; ORPO odds-ratio variant in `train_pref.py`. | $\beta = 0.10$ against frozen $\pi_{\text{SFT}}$ |
| **Offline Hybrid MPO** | `posttrain/mpo.py` | $w_p \mathcal{L}_{\text{DPO}} + w_q \mathcal{L}_{\text{BCO}}(\delta) + w_g \mathcal{L}_{\text{SFT}}$, where Binary Classifier Optimization (`BCO`) aligns unpaired lab successes/failures using an EMA `reward_shift` $\delta_{t+1} = m\delta_t + (1-m)\bar{r}_t$ ($m=0.95$). | $\beta = 0.10$ + `sft_weight = 0.20` |
| **Logit & On-Policy Distillation (`KD` / `TROPD`)** | `scientist/distill.py`, `posttrain/distill_opd.py` | Offline completion-masked Forward/Reverse/Mixed KL + top-$k$/X-Token projection (`distill_step_loss`); online Trust-Region Multi-Teacher OPD (`TROPD`, `arXiv:2601.02780`) mixing $A_{\text{TROPD},t} = \text{sg}[\log(\alpha\pi_T + (1-\alpha)\pi_\theta) - \log\pi_\theta] \ge \log(1-\alpha)$ with group task advantages. | `kl_coef = 0.005` (`k3`) + proximal $\alpha = 0.50$ |
| **Online PPO + VAPO** | `posttrain/ppo.py` | Clipped policy surrogate + clipped value loss (`0.5 * MSE`) + causal GAE ($\gamma=1.0, \lambda=0.95$), VAPO length-adaptive $\lambda_i = \max(0, 1 - \frac{1}{\alpha L_i})$ (`arXiv:2504.05118`), and positive-example NLL anchor (`positive_nll_coef`). | Detached `k3` KL subtracted in reward (`kl_coef = 0.04` / `0.02`) |
| **Online GRPO / GDPO / DAPO / GSPO / CISPO** | `posttrain/grpo.py`, `posttrain/losses.py`, `posttrain/advantages.py` | Group-mean or leave-one-out (`loo`) baseline; **GDPO** per-channel group standardization + weighted batch normalization; **DAPO** asymmetric clip-higher (`[0.20, 0.28]`), multi-round dynamic zero-variance resampling, overlong filtering, and soft overlong buffer; **ProRLv2.1** Z-MAD `<think>` length penalty + `stop_properly`; **GSPO** sequence-level geometric-mean ratios; **CISPO** detached clipped ratio weighting; **Ye et al.** dual-clip floor; **TIS / ICE-POP / Seq-Mask-TIS** off-policy importance weights. | Unbiased score-function Reverse-KL `reverse_kl_loss_term` (`k3`, `kl_coef = 0.01`) + `AdaptiveKLController` |

### Execution Status & Scientific Honesty
- **Completed Locally (CPU Smoke & Static Eval Summaries)**:
  - `scripts/smoke_all.py`: 15/15 Scientist and Operator jobs pass on CPU (`63.96s` total wall time on a 32-hidden byte-level GPT-2; `docs/smoke_results.md`). GRPO smoke receives `0.0` reward/loss because random weights do not emit valid JSON, reported transparently without faking numbers.
  - `posttrain` & `labrl` smoke tests: Exercise SFT, DPO, MPO, TROPD, PPO, GRPO/GDPO, recipe composition, and all 8 `VerifiedRewardTemplate` classes on CPU.
  - `results/protocol_reasoning/`: Contains static summary tables from an external OpenRouter evaluation of `Haiku 5.5` (`0.70` core macro acc), `GPT 6 Luna` (`0.65`), and `GLM 5.3 flash` (`0.38` at `12k` cap due to `54%` truncation; `0.52` on PQA+Cloning at `64k`) across `1,967` public LAB-Bench questions and `100`-item BioProBench greedy baselines.
- **Pending (Not Yet Executed)**:
  - **All single-GPU Colab A100 runs (`notebooks/colab_{scientist,operator,posttrain}.ipynb`, `configs/*_a100.yaml`) and multi-GPU NeMo-RL cluster jobs (`configs/posttrain/nemo_rl/`, `configs/recipes/`) are PENDING.** No trained scientific accuracy or hardware safety claim is made from CPU smoke checks.

---

## 5. History

| Commit | Date | Author | Branch / Milestone | Summary |
|---|---|---|---|---|
| `63697fb` | `2026-11-03` | `yaniguan` | `main` | Initial `v0 draft` repository scaffolding. |
| `c247afe` / `d44187d` | `2026-11-03` | `yaniguan` | `docs/benchmark-catalog-nov3` (PR #1) | Added curated 205-entry scientific & robotic benchmark catalog (`benchmarks/catalog.{csv,json,parquet}`) and schema validation test (`benchmarks/test_catalog.py`). |
| `b2bde46` | `2026-11-04` | `yaniguan` | `main` (`v0`) | Shipped `data/v0` (`49,139` provenance-labeled records, `20,517` robot frames), `src/labpt/` schema/splits/actions/constraints/logs, `scientist/` & `operator/` trainers, 15/15 CPU smoke report (`docs/smoke_results.md`), and A100 Colab notebooks. |
| `101933a` | `2026-11-10` | `yaniguan` | `main` (cf. `f02ba69`) | Added `labeval/` LAB-Bench evaluation harness, `results/protocol_reasoning/` model-comparison & token-budget studies, `common/llm.py` OpenRouter client, and `bioenvs/` (`PQA`, `ORD`, `ERR`, `GEN`) RLVR environments with exploit audits (`bioenvs/audit.py`). |
| `dbec763` | `2026-11-10` | `yaniguan` | `main` (cf. `fd97dea`) | Added standalone `posttrain/` stack (`sft`, `dpo`, `ppo`, `grpo`, `kl`, `reward_baselines`, `hacking_checks`, `contamination`, `eval_suite`) with SFT-anchored KL defaults, Shao et al. spurious-reward ladder, and LAB-Bench canary guards. |
| `34c1ccf` | `2026-11-11` | `yaniguan` | `feat/milestone1-posttrain-math-cuda` (`05664e9`) | **Milestone 1**: Fixed Reverse-KL pathwise vs score-function gradient (`reverse_kl_loss_term`), causal hidden-state shift in PPO GAE, DPO chosen-only KL metric, VAPO length-adaptive $\lambda$ & positive NLL, GSPO sequence ratios, TIS/ICE-POP/Seq-Mask-TIS IS weights, and Qwen/Llama/CUDA helpers. |
| `cdc897c` | `2026-11-11` | `yaniguan` | `feat/milestone2-gdpo-dapo-nemo-bridge` (`4dba632`) | **Milestone 2**: Upgraded `bioenvs` to `v2` anti-hack verifiers (`GEN` cluster dedup, `PQA` abstention, `ordered_action_reward`), multi-round DAPO dynamic resampling, ProRLv2.1 Z-MAD `<think>` length penalty, `src/labpt_nemo_rl/` dual-engine bridge, and 4-axis recipe YAML system. |
| `18b89b3` | `2026-11-11` | `yaniguan` | `feat/milestone3-mpo-and-tropd-distill` (`cedda0e`) | **Milestone 3**: Implemented hybrid offline `MPO` (`DPO + BCO + SFT` with EMA `reward_shift`), online `TROPD` multi-teacher trust-region distillation (`posttrain/distill_opd.py`), and completion-masked `scientist/distill.py` with cross-tokenizer safety checks. |
| `0a306cd` | `2026-11-11` | `yaniguan` | `feat/calisol23-conductivity-rlvr-suite` (`ecc99ae`, `3af068b`) | Integrated `CALiSol-23` experimental electrolyte conductivity dataset (`13,228` clean measurements across `27` DOIs), `calisol_design` & `calisol_predict` RLVR environments, `CALiSolEnv`, and `scripts/posttrain/calisol_pipeline.py`. |
| `6cb46a7` | `2026-11-11` | `yaniguan` | `feat/labrl-sdk-and-cli` (`1967db2`) | Packaged the repository into the installable `labrl` (`v0.2.0`) SDK & CLI (`pyproject.toml`, `PostTrainPipeline`, `SplitGuard`, `ScientificCorpus`, `TorchLocalEngine`, `NeMoRLEngine`, `VerifiableEnv`). |
| `94f0cf5` | `2026-11-12` | `yaniguan` | `feat/verified-reward-templates` (`8c04659`) | Added 8 anti-hack `VerifiedRewardTemplate` classes (`src/labrl/rewards/`), dimensional unit converter (`convert_scientific_unit`), pure-Python/RDKit SMILES verifier (`parse_smiles_invariants`), `GatedRewardTemplate`, and `CounterfactualPairScorer`. |

---

## 6. Unique Properties

### Architectural Invariants & Anti-Hack Guarantees
1. **Unbiased Score-Function Reverse-KL Surrogate (`posttrain/kl.py::reverse_kl_loss_term`)**: Directly differentiating Schulman's `k3` estimator $\frac{\pi_{\text{ref}}}{\pi} - \log\frac{\pi_{\text{ref}}}{\pi} - 1$ w.r.t. $\log\pi$ produces $1 - \frac{\pi_{\text{ref}}}{\pi}$, whose on-policy expectation equals the gradient of *Forward* KL $D_{\text{KL}}(\pi_{\text{ref}} \parallel \pi)$. Multiplying by the value-1 stop-gradient ratio $\exp(\log\pi - \text{sg}(\log\pi))$ restores the exact unbiased *Reverse*-KL gradient $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log\pi(a)\log\frac{\pi(a)}{\pi_{\text{ref}}(a)}]$.
2. **Orthogonal Multi-Channel GDPO Normalization (`posttrain/advantages.py::gdpo_advantages`)**: Standardizing each verifiable reward channel (`reward/conductivity`, `reward/support`, `reward/validity`, `reward/accuracy`, `reward/calibration`) within each prompt's rollout group before weighted summation prevents high-variance binary format bits from drowning subtle calibration or physical-support signals.
3. **Conservative Experimental Support Gating (`bioenvs/calisol.py`, `InverseDesignSupportTemplate`)**: Inverse formulation design multiplies predicted performance by a kernel density support gate (`exp(-d_min / support_scale)` or `1 - exp(-W_eff / tau)`), driving out-of-hull chemical hallucinations toward zero reward without relying on a learned neural reward model.
4. **Spurious-Reward Control Ladder & Runtime Hacking Monitor (`posttrain/reward_baselines.py`, `posttrain/hacking_checks.py`)**: Following Shao et al. (2026), every RLVR claim can be audited against `random`, `format_only`, `length`, `majority_vote`, and `incorrect_label` control arms alongside runtime detectors (`non_answering`, `constant_answer`, `position_format_bias`, `metagaming`, `canary_in_outputs`, and `proxy_gold_divergence`).
5. **Dual-Backend Portability (`TorchLocalEngine` $\leftrightarrow$ `NeMoRLEngine`)**: One `VerifiableEnv` / `VerifiedRewardTemplate` implementation runs both zero-GPU local PyTorch loops (`posttrain`) and Ray/vLLM/Megatron multi-node jobs (`yaniguan/RL`) via a loose-coupled `EnvironmentInterface` shim (`src/labpt_nemo_rl/interfaces.py`).

### Known Failure Modes & Audit Caveats
- **GDPO vs. Scalar Length-Shaping Interaction**: `shape_rewards` (`posttrain/rl_common.py`) appends a separate `"reward/length_shaping"` channel when `<think>` or overlong penalties fire; per-channel variance normalization in `gdpo_advantages` cancels out small `coef` scales unless `"reward/length_shaping"` is explicitly weighted or excluded from channel standardization.
- **Single-Reference Protocol Generation Ambiguity**: `GEN` (`bioenvs/gen.py`) aligns candidate protocols against a single reference procedure; scientifically valid alternative reagent orders or equivalent unit expressions can be under-rewarded.
- **Case-Collapsed Unit Normalization & String-Typed Safety Params**: `normalize_unit_symbol` lowercases unit strings (`nm` vs `nM`, `m` vs `M`), and `StructuredActionPlanTemplate` checks `safety_bounds` only on `(int, float)` values, requiring stricter type and case disambiguation before live robot gating.
- **Pending Empirical GPU Curves**: All learning-curve, KL-Pareto, and spurious-reward ladder comparisons on 1.7B–8B models remain unexecuted until A100/H100 runs are launched.

### Sibling-Repository Integration Contracts
- **[`RL`](https://github.com/yaniguan/RL) (`L3`)**: Consumes 4-axis composed NeMo-RL YAMLs (`labrl recipe --dump-yaml`) and `EnvironmentInterface.step()` (`BioProBenchEnv`, `CALiSolEnv`, `TemplateVerifiableEnv`).
- **[`YLM`](https://github.com/yaniguan/YLM) (`L0`/`L4`)**: Shares domain pretraining corpora boundaries and complements `StructuredActionPlanTemplate` / `PermutationOrderTemplate` with formal `LabLean` protocol verification.
- **[`WignerFlow`](https://github.com/yaniguan/WignerFlow) & [`assayfold`](https://github.com/yaniguan/assayfold) (`L0`/`L3`)**: Provide 3D equivariant molecular/protein observables verifiable via `ScalarObservableTemplate`, `MolecularValiditySimilarityTemplate`, and `SpectralPeakSetTemplate`.
- **[`twinstep`](https://github.com/yaniguan/twinstep) & [`labfill`](https://github.com/yaniguan/labfill) (`L3`)**: Align with `BioProBench` `PQA` (protocol parameter fill-in) and `ERR` / `ORD` (procedural step perturbation and error recovery).
- **[`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) & [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`)**: Enforce pre-execution state-machine and static verification downstream of `Operator` (`src/labpt/actions.py`, `src/labpt/constraints.py`) action plans.
- **[`lab-decide`](https://github.com/yaniguan/lab-decide) & [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (`L6`)**: Consume calibrated epistemic intervals/confidences (`CalibratedMCQTemplate`, `CALiSolPredict`) for trust-region abstention and scientific claim-rationale probing (`LitQA2`/`ProtocolQA`).
- **[`stepbot`](https://github.com/yaniguan/stepbot), [`lab-hands`](https://github.com/yaniguan/lab-hands), & [`lab-feet`](https://github.com/yaniguan/lab-feet) (`L1`/`L2`/`L5`)**: Execute verified `InstrumentAction` JSON calls and consume MCAP-style JSONL hardware traces (`src/labpt/logs.py`) to close the real-to-sim-to-RLVR loop via `run_log_to_reward_pairs`.
