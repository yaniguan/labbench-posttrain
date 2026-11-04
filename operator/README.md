# operator/ — the "hands"

Learn to *operate* a lab: turn protocols into robot actions, imitate real robot episodes,
prefer protocols that worked, and predict what happens next (a cheap world model).

| script | method | data |
|---|---|---|
| `../scripts/train_sft.py configs/operator/sft_*.yaml` | protocol -> structured actions SFT (LoRA/QLoRA) | ORD procedures, A-Lab run steps, robot instructions (`protocol2actions`, `safety_failure`, `outcome`) |
| `train_bc.py configs/operator/bc_*.yaml` | behavior cloning (state -> action MLP) | `data/v0/robot_frames.parquet` (DROID-100 + TOTO) |
| `../scripts/train_pref.py configs/operator/dpo_*.yaml` | DPO / ORPO on success vs failure | `pairs_*.jsonl` kinds `success_vs_failure`, `corrupted_actions` |
| `../scripts/train_rm.py configs/operator/rm_*.yaml` | reward model from execution success | same pairs |
| `../scripts/train_grpo.py configs/operator/grpo_*.yaml` | GRPO, verifiable rewards | `protocol2actions` + `provenance` |
| `world_model.py configs/operator/wm_*.yaml` | world model: (state, action) -> next state, multi-step rollouts | `robot_frames.parquet` |

All training scripts accept `--smoke` for tiny CPU runs on real local data. Run the complete suite with `bash scripts/smoke_all.sh` from the repo root.

`check_constraints.py --plan PLAN.json --constraints CONSTRAINTS.json` scores high-level skills/waypoints and extra avoid-box constraints against trusted workspace bounds, joint limits and swept-point collision boxes. The returned `reward` is 0 or 1. This is not full IK or arm collision checking. Plan and constraint formats are `labpt.constraints::{OperatorPlan,ConstraintSpec}`.

`labpt.actions` defines strict instrument JSON schemas, Data/Error + skip/retry results and run-log T1 rewards/recovery pairs. `labpt.logs` supplies MCAP-style JSONL envelopes without a binary MCAP dependency. Existing coarse SFT labels and low-dimensional BC actions are not yet converted into complete executable skill/waypoint plans.

See [the shared design](../docs/DESIGN.md). GPU results: **PENDING (A100 Colab)**.
