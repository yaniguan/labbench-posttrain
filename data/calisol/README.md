# CALiSol-23 Electrolyte Conductivity Dataset

Official experimental lithium-ion liquid electrolyte ionic conductivity dataset (`13,825` raw rows; `13,269` strictly positive finite measurements across `27` peer-reviewed publications, `13` lithium salts, and `38` organic solvents).

## Citation & License
- **Paper / Dataset**: P.-P. De Breuck, M. L. Evans, G.-M. Rignanese, *CALiSol-23: Experimental electrolyte conductivity dataset for lithium-ion batteries*, DTU Data (2024), DOI: `10.11583/DTU.c.6929599.v1` (CC-BY 4.0).
- **Files**:
  - `calisol23.csv`: `doi`, `k` ($\text{mS/cm}$), `T` ($\text{K}$), `c`, `salt`, `c units`, `solvent ratio type`, and 38 solvent fraction columns.
  - `calisol_smiles.csv`: Canonical names, abbreviations, and SMILES strings for all 13 salts and 38 solvents.
- **Refresh script**: `python scripts/get_calisol.py`
