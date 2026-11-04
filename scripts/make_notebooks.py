"""Generate A100 Colab notebooks from the current source/data snapshot; no uploads or git commands."""
import json
from pathlib import Path

SETUP = [
    "from pathlib import Path\nimport os, subprocess, sys\n"
    "repo = Path('/content/labbench-posttrain')\n"
    "assert (repo / 'data/v0/records.parquet').is_file(), 'Unpack the current repo snapshot with data/v0 at /content/labbench-posttrain first.'\n"
    "os.chdir(repo)\n"
    "subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '.venv'], check=True)\n"
    "PY = str(repo / '.venv/bin/python')\n"
    "subprocess.run([PY, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)",
    "import torch\nassert torch.cuda.is_available(), 'Select Runtime > Change runtime type > A100.'\n"
    "assert 'A100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)\n"
    "torch.backends.cuda.matmul.allow_tf32 = True\n"
    "os.environ['NVIDIA_TF32_OVERRIDE'] = '1'\n"
    "print(subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout)\n"
    "def run(*args):\n    subprocess.run([PY, *args], check=True)",
    "# Verify data contracts and refresh reports before long jobs.\n"
    "run('-m', 'pytest', '-q')\nrun('scripts/assay_quality.py')\nrun('scripts/benchmark_link.py')",
]
TRACKS = {
    'scientist': [
        ('Representations', 'scientist/train_encoders.py', 'encoders'),
        ('Modality/text alignment', 'scientist/train_align.py', 'align'),
        ('Molecule soft-token projector + LoRA', 'scientist/train_projector.py', 'projector'),
        ('SFT with LoRA', 'scripts/train_sft.py', 'sft'),
        ('DPO', 'scripts/train_pref.py', 'dpo'),
        ('Reward model', 'scripts/train_rm.py', 'rm'),
        ('GRPO with T1/T2 rewards and KL', 'scripts/train_grpo.py', 'grpo'),
        ('Teacher/student distillation', 'scientist/distill.py', 'distill'),
    ],
    'operator': [
        ('SFT with LoRA', 'scripts/train_sft.py', 'sft'),
        ('DPO', 'scripts/train_pref.py', 'dpo'),
        ('ORPO', 'scripts/train_pref.py', 'orpo'),
        ('Reward model', 'scripts/train_rm.py', 'rm'),
        ('GRPO with T1/T2 rewards and KL', 'scripts/train_grpo.py', 'grpo'),
        ('Behavior cloning', 'operator/train_bc.py', 'bc'),
        ('World model and held-out rollouts', 'operator/world_model.py', 'wm'),
    ],
}


def code(source):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)}


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


for track, jobs in TRACKS.items():
    cells = [markdown(f"# {track.title()} track: full A100 jobs\n\nGPU results: **PENDING (A100 Colab)**. These cells have not been run on a GPU here.\n\n"
                      "Unpack a snapshot of the current workspace (code, configs, tests, benchmarks, data/v0) at `/content/labbench-posttrain`. "
                      "Exclude data/raw, .git, .venv and outputs. Do not use an older remote copy. Run cells top to bottom on an A100. "
                      "Dependencies are installed into a venv that reuses Colab CUDA torch. No HuggingFace uploads.\n\n"
                      "Each method is an independent baseline initialized from the model in its config. DPO/GRPO do not automatically continue the SFT adapter. "
                      "For a sequential experiment, merge the SFT adapter locally and explicitly override model=<merged directory>. "
                      "T3 assay calibration and live blind experiments remain TODO. Long jobs may exceed a single Colab session; retain outputs on Drive if needed.")]
    cells += [code(s) for s in SETUP]
    for title, script, method in jobs:
        cells += [markdown('## ' + title), code(f"run('{script}', 'configs/{track}/{method}_a100.yaml')")]
    cells += [markdown('## Held-out SFT evaluation'), code(
        f"run('scripts/eval_llm.py', '--model', 'Qwen/Qwen2.5-7B-Instruct', '--adapter', 'outputs/a100/{track}_sft/adapter', "
        f"'--n', '200', '--out', 'outputs/a100/eval_{track}_sft.json')")]
    cells += [markdown('## Time split and blind reservation (offline rehearsal)'), code(
        "# Builds a separate dataset; full jobs above use the checked-in group split.\n"
        "run('scripts/build_dataset.py', '--out', 'outputs/time_split', '--split-mode', 'time', '--val-year', '2009', '--test-year', '2011', "
        "'--prospective-source', 'platero_cytation3') if Path('data/interim').is_dir() else print('Optional split rebuild needs data/interim; do not download or modify data/raw.')\n"
        "# To train on this rebuild, generate tasks with --data-dir outputs/time_split, then override train_file/eval_file.\n"
        "# prospective_blind.parquet stays out of routine task generation; lab staff must seal future results.")]
    cells += [code("import json\nfor path in sorted(Path('outputs/a100').glob('**/profile/summary.json')):\n    print(path, json.loads(path.read_text()))")]
    notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100'},
                'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    Path(f'notebooks/colab_{track}.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
print('updated A100 notebooks; execution results PENDING')
