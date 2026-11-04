"""Per-frame (state, action, next_state) table for behavior cloning + world model.
Reads the same small LeRobot parquet shards as ingest_robot.py (HF cache). Output: data/v0/robot_frames.parquet"""
import hashlib

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPOS = ["lerobot/droid_100", "lerobot/toto"]
MAX_FRAMES = 20000
rows = []
for repo in REPOS:
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))[:1]
    df = pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in files])
    n = 0
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        s = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        a = np.stack(g["action"].to_numpy()).astype(np.float32)
        h = int(hashlib.md5(f"{repo}|{ep}".encode()).hexdigest(), 16) % 100
        split = "test" if h < 10 else "val" if h < 20 else "train"  # split by episode
        for t in range(len(g) - 1):
            rows.append({"repo": repo, "episode": int(ep), "t": t, "split": split,
                         "state": s[t].round(5).tolist(), "action": a[t].round(5).tolist(), "next_state": s[t + 1].round(5).tolist()})
            n += 1
        if n >= MAX_FRAMES // len(REPOS):
            break
    print(repo, n)
out = pd.DataFrame(rows)
out.to_parquet("data/v0/robot_frames.parquet", index=False)
print(out.split.value_counts().to_dict(), "state_dim", len(rows[0]["state"]))
