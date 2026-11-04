# Data licenses

Code: MIT. Data: each record keeps the license of its source in the `license` field.
Share-alike sources (CC-BY-SA) mean that a redistributed *derived dataset* of those records must use the same license.
If you publish `data/v0` as one file, the strictest terms apply to those rows (filter by `license` if needed).

| source | license (recorded) | notes |
|---|---|---|
| Open Reaction Database (ord-data) | CC-BY-SA-4.0 | attribution + share-alike |
| ChEMBL (API) | CC-BY-SA-3.0 | attribution + share-alike |
| PubChem BioAssay / Compound | public domain per NCBI policy | some depositors add terms; check per AID before commercial use |
| QM9 (Ramakrishnan et al. 2014, figshare) | CC0-1.0 | mirror used: DeepChem S3 csv |
| Open X-Embodiment via LeRobot (DROID-100, TOTO, Berkeley UR5, TACO Play, NYU Franka Play) | CC-BY-4.0 (upstream OXE / DROID) | HF repo tags differ for some (`mit` for droid_100/toto/nyu); we record the upstream CC-BY-4.0 and keep the HF tag in `conditions.hf_license_tag` |
| Ada SDL campaign log (berlinguette/ada) | MIT | |
| A-Lab Precursor Genome ledger + raw XRD | CC-BY-4.0 | |
| ChemOS 2.0 optics log | CC-BY-4.0 | |
| RODARE record 4101 (NMR/HPLC/IR of cmpd 10a) | CC-BY-4.0 | |
| DaRUS hydrolase HPLC (doi:10.18419/darus-4052) | CC-BY-4.0 | |
| PLATERO Cytation3 plate (Zenodo 7224497) | CC-BY-4.0 | |
| Floral bacteria growth curves (Zenodo 4304841) | CC0-1.0 | |
| Ca-imaging ELN RO-Crate (SFB-ELAINE) | CC-BY-4.0 | |
| Alemany et al. microscopy (Zenodo 15682874) | CC-BY-4.0 | 64 patches only |

Not used: protocols.io (API needs an access token), Materials Project (API key), any SES AI data.
