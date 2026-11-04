"""Ingest small robot-manipulation subsets (Open X-Embodiment / DROID via LeRobot on HF).

We only download metadata + low-dim action parquet files (no videos).
One record per episode: language instruction, action summary, coarse plan.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.protocol_parser import robot_plan_from_instruction
from labpt.schema import Outcome, Record, make_id

# (hf repo, upstream license we record, robot, max data files)
DATASETS = [
    ("lerobot/droid_100", "CC-BY-4.0", "Franka Panda (DROID)", 1),
    ("lerobot/toto", "CC-BY-4.0", "Franka Panda (TOTO pouring/scooping)", 1),
    ("lerobot/berkeley_autolab_ur5", "CC-BY-4.0", "UR5", 1),
    ("lerobot/taco_play", "CC-BY-4.0", "Franka Panda (TACO play)", 1),
    ("lerobot/nyu_franka_play_dataset", "CC-BY-4.0", "Franka Panda (NYU play)", 1),
]

out = []
for repo, lic, robot, max_files in DATASETS:
    api = HfApi()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")]
    hf_tag = (api.dataset_info(repo).card_data or {}).get("license")
    info = json.load(open(hf_hub_download(repo, "meta/info.json", repo_type="dataset")))
    tasks = None
    if "meta/tasks.parquet" in files:
        tasks = pd.read_parquet(hf_hub_download(repo, "meta/tasks.parquet", repo_type="dataset"))
    elif "meta/tasks.jsonl" in files:
        tasks = pd.read_json(hf_hub_download(repo, "meta/tasks.jsonl", repo_type="dataset"), lines=True)
    task_map = {}
    if tasks is not None:
        if "task_index" in tasks.columns:
            tcol = "task" if "task" in tasks.columns else None
            task_map = dict(zip(tasks["task_index"], tasks[tcol] if tcol else tasks.index))
        else:  # v3: index is task string
            task_map = dict(zip(tasks.iloc[:, 0], tasks.index))
    data_files = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))[:max_files]
    df = pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in data_files])
    fps = info.get("fps")
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index") if "frame_index" in g else g
        acts = np.stack(g["action"].to_numpy())
        ti = int(g["task_index"].iloc[0]) if "task_index" in g else None
        instr = str(task_map.get(ti, "")).strip()
        grip = acts[:, -1]
        thr = (grip.max() + grip.min()) / 2
        toggles = int(np.sum(np.abs(np.diff((grip > thr).astype(int)))))
        summary = {
            "n_frames": int(len(g)),
            "duration_s": round(len(g) / fps, 2) if fps else None,
            "action_dim": int(acts.shape[1]),
            "action_mean": [round(float(x), 4) for x in acts.mean(0)],
            "action_std": [round(float(x), 4) for x in acts.std(0)],
            "gripper_toggles": toggles,
        }
        r = Record(
            id=make_id("robot", repo, ep),
            source=f"open_x_embodiment/{repo.split('/')[1]}",
            source_id=f"{repo}#episode{ep}",
            source_url=f"https://huggingface.co/datasets/{repo}",
            license=lic,
            modality="robot_episode",
            provenance="measurement",
            method="teleoperated demonstration",
            instrument=robot,
            property="episode_duration",
            value=summary["duration_s"],
            units="s",
            entity={"instruction": instr},
            conditions={"fps": fps, "hf_license_tag": hf_tag, "action_summary": summary},
            protocol_text=instr or None,
            protocol_steps=robot_plan_from_instruction(instr),
            # LeRobot ports of these sets contain (mostly) successful demos; no per-episode label
            outcome=Outcome(success=None, metric="demo_completed"),
            group_key=f"{repo}|{instr.lower()}",
        )
        out.append(annotate(r))
    print(repo, "episodes:", df["episode_index"].nunique())

print("robot records:", write_jsonl(out, Path("data/interim/robot.jsonl")))
