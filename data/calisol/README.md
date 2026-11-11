# CALiSol-23 Electrolyte Conductivity Dataset

Official experimental lithium-ion liquid electrolyte ionic conductivity dataset (`13,825` raw rows from the public CSV; `13,228` clean measurements after this repo's hygiene filters, across `27` peer-reviewed publications, `13` salt labels, and `38` organic solvents). The official paper describes `14` lithium salts; this CSV uses `13` salt labels (`LiTFSI` and `LiN(CF3SO2)2` are kept as separate labels).

## Citation & License
- **Paper**: Paolo de Blasio, Jonas Elsborg, Tejs Vegge, Eibar Flores, and Arghya Bhowmik, *CALiSol-23: Experimental electrolyte conductivity data for various Li-salts and solvent combinations*, Scientific Data 11, 750 (2024), DOI: `10.1038/s41597-024-03575-8`.
- **Dataset**: DTU Data, DOI: `10.11583/DTU.c.6929599.v1` (CC-BY 4.0). Code mirror: `https://github.com/Pele0599/CALiSol-23`.
- **Files**:
  - `calisol23.csv`: `doi`, `k` ($\text{mS/cm}$), `T` ($\text{K}$), `c`, `salt`, `c units`, `solvent ratio type`, and 38 solvent fraction columns.
  - `calisol_smiles.csv`: Canonical names, abbreviations, and SMILES strings for the salt labels and 38 solvents.
- **Refresh script**: `python scripts/get_calisol.py`
