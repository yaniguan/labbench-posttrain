# labbench-posttrain

A small lab post-training project with two tracks:

- **Scientist:** learn scientific representations, align modalities with text, answer historical questions with evidence labels, and run SFT, DPO, reward modeling, GRPO and distillation.
- **Operator:** turn protocols into actions, learn state/action behavior and a world model, and score structured plans against simple physical constraints.

**GPU results: PENDING (A100 Colab).** CPU smoke checks verify code execution. They do not establish useful scientific accuracy or safe robot control.

See [the design mapped to code](docs/DESIGN.md), [dataset overview](docs/dataset_overview.md), [benchmark map](docs/benchmark_map.md), [assay quality](docs/assay_quality.md), and [CPU smoke results](docs/smoke_results.md). Each design component is marked built, partial or TODO.

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
