# Benchmark catalog for lab-automation post-training

205 datasets for scientific AI: protocols, lab robotics, reactions/HTE, assays, QM/MLIP, spectra, OCSR, science QA/agents, proteins, batteries.

Files: `catalog.csv`, `catalog.parquet`, `catalog.json`, `chart_*.png`.

Columns: name, domain, tasks, modality, scale, source, physical_scale, license, access, link, paper, notes, relevance, track_fit (scientist / operator / both), messiness (raw / semi-clean / clean), link_http, verified.

How it was made: first draft by the codex CLI (model gpt-6.1-sol, reasoning effort xhigh, with web search). Then 15 raw / messy data sources were added by hand. All links were checked with curl on 2026-11-03. `verified` = link_ok, bot_blocked_403 (site blocks scripts, page likely fine) or unverified_link. Scale and license values come from the draft and are not all checked by hand. Treat them as hints.


## Counts by modality
- 3D structure/trajectory: 50
- tabular assay: 36
- multimodal: 28
- spectra: 23
- text/protocol: 17
- robot action/video: 16
- image: 16
- time-series instrument log: 11
- graph: 8

## Counts by source
- experimental measurement: 78
- mixed: 61
- first-principles compute: 38
- human annotation: 27
- ML prediction: 1

## Counts by physical_scale
- molecular: 100
- lab-process: 45
- material: 26
- cell: 18
- device: 11
- organism: 5

## Counts by access
- open: 191
- registration: 8
- restricted: 6

## Counts by track_fit
- scientist: 141
- both: 48
- operator: 16

## Counts by messiness
- semi-clean: 155
- raw: 25
- clean: 25

## Counts by verified
- link_ok: 171
- unverified_link: 21
- bot_blocked_403: 13

## Top 30 picks for post-training
| # | Dataset | Why |
|---|---|---|
| 1 | protocols.io public protocols | SFT protocol->actions |
| 2 | BioProt | protocol->pseudocode SFT/eval |
| 3 | ProtocolQA | protocol error-finding eval |
| 4 | LAB-Bench | lab reasoning eval |
| 5 | BioProBench | protocol RL/eval |
| 6 | Open Reaction Database (ORD) | reaction outcomes + conditions, provenance |
| 7 | Buchwald-Hartwig HTE (Ahneman) | yield prediction reward |
| 8 | Suzuki-Miyaura HTE (Perera) | condition optimisation reward |
| 9 | HiTEA | HTE incl. negative results |
| 10 | AstraZeneca ELN Buchwald-Hartwig | real ELN, messy yields |
| 11 | Dark Reactions Project | failed runs -> outcome prediction |
| 12 | USPTO-condition | condition recommendation |
| 13 | ORDerly | clean reaction benchmark splits |
| 14 | Open X-Embodiment | operator pretraining |
| 15 | DROID | teleop manipulation |
| 16 | LabDex | lab-specific robot data |
| 17 | JUMP Cell Painting | cell image outcome readout |
| 18 | RxRx3 | perturbation phenotypes |
| 19 | Polaris | curated eval with leakage-aware splits |
| 20 | Papyrus | clean bioactivity |
| 21 | PubChem BioAssay | raw HTS incl. inactives |
| 22 | Chemotion Repository | ELN + raw analytics |
| 23 | nmrXiv | raw NMR interpretation |
| 24 | MassSpecGym | MS interpretation eval |
| 25 | Multimodal Spectroscopic Dataset | spectra->structure |
| 26 | OMol25 | MLIP / QM tool reward |
| 27 | Matbench Discovery | materials eval |
| 28 | Severson fast-charging battery dataset | time-series outcome prediction |
| 29 | Attia closed-loop fast-charging dataset | closed-loop optimisation logs |
| 30 | MaCBench | multimodal chem lab eval |

## Gaps
- Very little public data of failed runs, raw instrument dumps, or liquid-handling robot actions.
- No standard benchmark that pairs a protocol with runnable robot code.
- Many licenses are unknown or vary per record. Check before use.
