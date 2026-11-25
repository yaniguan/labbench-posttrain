"""Deterministic group splits, chronological splits, canonical scaffold keys, and a sealed holdout."""
import functools
import hashlib
import math

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
    _LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
    _UNCHARGER = rdMolStandardize.Uncharger()
    _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
except ImportError:  # pragma: no cover - optional in minimal non-chemistry installs
    _RDKIT_AVAILABLE = False


@functools.lru_cache(maxsize=65536)
def canonical_scaffold_group_key(smiles: str | None) -> str:
    """Compute leak-free standardized molecular `group_key` (#46).

    1. Strips salt counterions and solvent co-formers (`LargestFragmentChooser`).
    2. Clears isotopes, removes stereochemistry, neutralizes formal charges (`Uncharger`),
       and canonicalizes keto-enol / lactam-lactim tautomers & Kekule forms
       (`TautomerEnumerator().Canonicalize`).
    3. Extracts canonical tautomer-normalized `MurckoScaffold` (`scaffold|<smiles>`).
    4. Never collapses acyclic compounds onto `'scaffold|'`: falls back to
       `acyclic|<canonical_parent_smiles>` so acyclic molecules distribute 80/10/10
       across `train`/`val`/`test`.
    """
    raw = (smiles or "").strip()
    if not raw:
        return "invalid|empty"
    if not _RDKIT_AVAILABLE:
        largest = max((frag.strip() for frag in raw.split(".") if frag.strip()), key=len, default=raw)
        clean = largest.replace("@", "").replace("/", "").replace("\\", "")
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]
        return f"acyclic|{digest}"

    mol = Chem.MolFromSmiles(raw)
    if mol is None or mol.GetNumAtoms() == 0:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"invalid|{digest}"

    mol = _LARGEST_FRAGMENT_CHOOSER.choose(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    Chem.RemoveStereochemistry(mol)
    mol = _UNCHARGER.uncharge(mol)
    mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)

    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is not None and scaf.GetNumAtoms() > 0:
        scaf = _TAUTOMER_ENUMERATOR.Canonicalize(scaf)
        scaf_smi = Chem.MolToSmiles(scaf, isomericSmiles=False, canonical=True)
        if scaf_smi:
            return f"scaffold|{scaf_smi}"

    parent_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    if parent_smi:
        return f"acyclic|{parent_smi}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"acyclic|{digest}"


def split_of(group: str) -> str:
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "val" if h < 20 else "train"


def document_year(record):
    year = record.conditions.get("document_year")
    if isinstance(year, bool) or year is None:
        return None
    try:
        value = float(year)
        return int(value) if math.isfinite(value) and value.is_integer() else None
    except (TypeError, ValueError):
        return None


def assign_splits(records, split_mode="group", val_year=None, test_year=None,
                  prospective_sources=(), prospective_ids=(), prospective_fraction=0.0):
    """Mutate Record.split and return the exact split policy.

    Time splits use record years (scaffolds may recur across years). Undated
    records use group hashing. Blind selection takes whole leakage groups,
    before year thresholds are computed, and never enters train/val/test.
    This offline reservation is not evidence of a live prospective test.
    """
    if split_mode not in ("group", "time"):
        raise ValueError("split_mode must be group or time")
    if not 0 <= prospective_fraction < 1:
        raise ValueError("prospective_fraction must be in [0, 1)")
    sources, ids = set(prospective_sources), set(prospective_ids)
    unknown = ids - {r.id for r in records}
    if unknown:
        raise ValueError(f"unknown prospective ids: {sorted(unknown)[:5]}")
    if sources - {r.source for r in records}:
        raise ValueError("unknown prospective source")
    group = lambda r: r.group_key or r.id
    blind = {group(r) for r in records if r.source in sources or r.id in ids}
    for r in records:
        h = int(hashlib.sha256(("blind|" + group(r)).encode()).hexdigest(), 16) / 2**256
        if h < prospective_fraction:
            blind.add(group(r))
    years = sorted({document_year(r) for r in records if group(r) not in blind and document_year(r) is not None})
    if split_mode == "time":
        if (val_year is None) != (test_year is None):
            raise ValueError("provide both val_year and test_year")
        if val_year is not None and val_year >= test_year:
            raise ValueError("val_year must be before test_year")
    if split_mode == "time" and years:
        if val_year is None:
            if len(years) < 3:
                raise ValueError("time split needs three years or explicit cutoffs")
            val_year = years[min(len(years) - 2, int(len(years) * 0.8))]
            test_year = years[min(len(years) - 1, max(years.index(val_year) + 1, int(len(years) * 0.9)))]
        if val_year >= test_year:
            raise ValueError("val_year must be before test_year")
    undated = 0
    for r in records:
        year = document_year(r)
        if group(r) in blind:
            r.split = "prospective_blind"
        elif split_mode == "time" and year is not None:
            r.split = "test" if year >= test_year else "val" if year >= val_year else "train"
        else:
            undated += split_mode == "time"
            r.split = split_of(group(r))
    return {"split_mode": split_mode, "val_year": val_year, "test_year": test_year,
            "undated_group_fallback_records": undated, "prospective_groups": len(blind),
            "prospective_sources": sorted(sources), "prospective_fraction": prospective_fraction,
            "prospective_status": "reserved offline; live blind lab evaluation TODO"}
