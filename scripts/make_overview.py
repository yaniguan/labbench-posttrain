"""Write docs/dataset_overview.md + docs/img/*.png charts from the real data/v0 files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

V0, DOCS = Path("data/v0"), Path("docs")
(DOCS / "img").mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(V0 / "records.parquet")
card = json.load(open(V0 / "datacard.json"))
tasks = json.load(open(V0 / "tasks/stats.json"))


def bar(col, fname, title):
    vc = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(vc))))
    ax.barh(vc.index[::-1].astype(str), vc.values[::-1], color="#4477aa")
    for i, v in enumerate(vc.values[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("records")
    fig.tight_layout()
    fig.savefig(DOCS / "img" / fname, dpi=110)
    plt.close(fig)


bar("source", "by_source.png", "Records by source")
bar("provenance", "by_provenance.png", "Records by provenance")
bar("modality", "by_modality.png", "Records by modality")
bar("physical_scale", "by_scale.png", "Records by physical scale")
bar("license", "by_license.png", "Records by license")

# one raw spectrum vs cleaned
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, src in zip(axes, ["alab_xrd_raw", "darus_hydrolase_hplc", "growth_curves_cytation5"]):
    r = df[df.source == src].iloc[0]
    c = json.loads(r.conditions)
    x, y = (c.get("x"), c.get("y")) if c.get("y") else (c.get("t_min"), c.get("od600"))
    ax.plot(x, y, lw=1)
    ax.set_title(f"{src}\n{r.source_id[:40]}", fontsize=8)
fig.tight_layout()
fig.savefig(DOCS / "img" / "curves.png", dpi=110)
plt.close(fig)


def table(d):
    return "\n".join(["| key | records |", "|---|---:|"] + [f"| {str(k).replace(chr(124), ' / ')} | {v} |" for k, v in d.items()])


ct = pd.crosstab(df.source, df.provenance)
ct_md = "| source | " + " | ".join(ct.columns) + " | license |\n|---|" + "---:|" * len(ct.columns) + "---|\n"
for s, row in ct.iterrows():
    ct_md += f"| {s} | " + " | ".join(str(v) for v in row.values) + f" | {df[df.source == s].license.iloc[0]} |\n"
ms = pd.crosstab(df.modality, df.physical_scale)
ms_md = "| modality | " + " | ".join(ms.columns) + " |\n|---|" + "---:|" * len(ms.columns) + "\n"
for s, row in ms.iterrows():
    ms_md += f"| {s} | " + " | ".join(str(v) for v in row.values) + " |\n"


def raw_vs_clean(source, raw_lines=12, trim=1600):
    r = df[df.source == source].iloc[0]
    raw_path = r.raw if isinstance(r.raw, str) else None
    raw = f"(clean API/benchmark source; original at {r.source_url})"
    if raw_path and Path(raw_path).exists() and Path(raw_path).suffix not in (".xlsx", ".png"):
        raw = "".join(open(raw_path, errors="replace").readlines()[:raw_lines])[:1500]
    elif raw_path:
        raw = f"(binary file kept at {raw_path})"
    rec = {k: r[k] for k in ["id", "source", "license", "modality", "provenance", "physical_scale", "method", "property", "value", "units"]}
    for k in ["entity", "conditions", "protocol_steps", "outcome", "annotations"]:
        rec[k] = json.loads(r[k])
    if "x" in rec["conditions"]:
        rec["conditions"]["x"] = rec["conditions"]["x"][:5] + ["..."]
        rec["conditions"]["y"] = rec["conditions"]["y"][:5] + ["..."]
    txt = json.dumps(rec, indent=1, default=str)[:trim]
    return f"### {source}\n\n**Raw (messy) excerpt** — `{raw_path}`\n\n```\n{raw}\n```\n\n**Cleaned record**\n\n```json\n{txt}\n```\n"


md = f"""# Dataset overview (v0) — the real training data

All numbers below are computed from `data/v0/records.parquet` by `scripts/make_overview.py`.
Nothing here is synthetic: every record comes from a public source and points back to it (`source_url`).

- **records:** {card['n_records']}  (invalid dropped: {card['n_invalid_dropped']}, duplicates dropped: {card['n_duplicates_dropped']})
- **splits:** {card['by_split']}  — split by leakage group (molecule scaffold, ORD dataset, robot task, A-Lab element system, ...); groups that appear in more than one split: **{card['groups_in_multiple_splits']}**
- **annotators:** {card['annotators']}  (`rule:v0` and `llm:*` are automatic, `human:*` came from the source authors)

![by source](img/by_source.png)

## Source x provenance (with license)

{ct_md}
![provenance](img/by_provenance.png)

## Modality x physical scale

{ms_md}
![modality](img/by_modality.png) ![scale](img/by_scale.png)

## Licenses

{table(card['by_license'])}

![license](img/by_license.png)

## Task examples and preference pairs built from it

{table(tasks['examples'])}

{table(tasks['pairs'])}

## Raw vs cleaned: real examples

We keep both. Raw excerpts live in `data/v0/raw_samples/` (small copies or first lines of the original file).
Full raw files are re-downloaded by `scripts/ingest_messy.py` into `data/raw/` (not in git).

![curves](img/curves.png)

"""
for src in ["ada_sdl_log", "alab_precursor_genome", "alab_xrd_raw", "chemos2_log", "rodare_cmpd10a", "ir_sample", "platero_cytation3",
            "growth_curves_cytation5", "eln_ca_imaging", "ord", "open_x_embodiment/toto", "chembl", "qm9"]:
    if src in set(df.source):
        md += raw_vs_clean(src) + "\n"

md += "\n## Browse more\n\n`data/v0/sample.jsonl` has 20 full records per source (one JSON per line). Quick look:\n\n"
md += "```python\nimport pandas as pd\ndf = pd.read_parquet('data/v0/records.parquet')\ndf.groupby(['source','provenance']).size()\n```\n"
(DOCS / "dataset_overview.md").write_text(md)
print("wrote docs/dataset_overview.md", len(md))
