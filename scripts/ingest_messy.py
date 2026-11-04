"""Ingest MESSY real lab data (raw instrument files, run logs, failed runs, notebook text, images).

Sources were found by the codex CLI scout (gpt-6.1-sol, reasoning=ultra) and checked by hand.
Every file gets one small parse function (KISS). We keep:
  - the raw/messy excerpt  -> data/v0/raw_samples/<file>   (small copies or first lines)
  - the cleaned records    -> data/interim/messy.jsonl     (record.raw points at the raw excerpt)
"""
import datetime
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.protocol_parser import parse_procedure
from labpt.schema import Action, Annotation, Outcome, Record, make_id

RAW = Path("data/raw/messy")
SAMPLES = Path("data/v0/raw_samples")
SAMPLES.mkdir(parents=True, exist_ok=True)
MAN = json.load(open(RAW / "manifest.json"))
out: list[Record] = []


def lic(f):
    return MAN[f]["license"]


def url(f):
    return MAN[f]["direct_download_url"]


def keep_head(f, n_lines=200):
    """Save the first lines of a text file as the raw sample."""
    dst = SAMPLES / (f + ".head.txt")
    with open(RAW / f, errors="replace") as src:
        dst.write_text("".join(next(src, "") for _ in range(n_lines)))
    return str(dst)


def downsample(x, y, n=256):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x) == 0:
        return [], []
    grid = np.linspace(x[0], x[-1], n)
    return [round(float(v), 5) for v in grid], [round(float(v), 5) for v in np.interp(grid, x, y)]


# ---------------------------------------------------------------- 1. Ada SDL log
def ada_log():
    f = "ada_campaign2.log"
    raw = keep_head(f, 300)
    pat = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL):([^:]+):(.*)$")
    counts, kept = {}, 0
    for i, line in enumerate(open(RAW / f, errors="replace")):
        m = pat.match(line.strip())
        if not m:
            continue
        level, logger, msg = m.groups()
        counts[level] = counts.get(level, 0) + 1
        is_fail = level in ("ERROR", "CRITICAL", "WARNING") or "exception" in msg.lower() or "timeout" in msg.lower()
        # keep all failure-ish events (capped) and every 400th normal event
        if (is_fail and kept < 1500) or (i % 400 == 0):
            kind = "instrument_timeout" if "TMO" in msg or "imeout" in msg else ("software_error" if is_fail else None)
            r = Record(
                id=make_id("ada", i), source="ada_sdl_log", source_id=f"campaign2:line{i}", source_url=url(f),
                license=lic(f), modality="run_log", provenance="measurement", physical_scale="lab-process",
                method="self-driving lab controller log", instrument="Ada thin-film SDL (UBC)",
                entity={"logger": logger.strip()}, conditions={"level": level, "line_no": i},
                protocol_text=msg.strip()[:500], raw=raw,
                outcome=Outcome(success=None if not is_fail else False, metric="event_ok"),
                group_key="ada|campaign2",
            )
            if kind:
                r.annotations.append(Annotation(key="failure_reason", value=kind, annotator="rule:v0"))
            out.append(r)
            kept += is_fail
    print("ada", counts)


# ---------------------------------------------------------------- 2. A-Lab ledger
def alab_ledger():
    f = "alab_ledger.json"
    d = json.load(open(RAW / f))
    sample0 = dict(d["samples"][10])
    (SAMPLES / "alab_ledger.one_sample.json").write_text(json.dumps(sample0, indent=1)[:200000])
    raw = str(SAMPLES / "alab_ledger.one_sample.json")
    n = 0
    for s in d["samples"]:
        syn = s.get("synthesis") or {}
        oc = s.get("outcome") or {}
        if not syn:
            continue
        cat = oc.get("reaction_category")
        prof = syn.get("temperature_profile") or {}
        tmin, tc = prof.get("time_minutes") or [], prof.get("temperature_celsius") or []
        precs = [p.get("formula") for p in s.get("precursors", [])]
        tgt = {m["name"]: m["target_mass"] for m in syn.get("precursor_target_masses") or []}
        act = {m["name"]: m["actual_mass"] for m in syn.get("precursor_actual_masses") or []}
        dose_err = {k: round(act[k] - tgt[k], 5) for k in tgt if k in act}
        steps = [Action(action="add", params={"what": k, "target_g": tgt[k], "actual_g": act.get(k)}) for k in tgt]
        steps += [Action(action="stir", params={"what": "mix powders"}),
                  Action(action="heat", params={"temperature_C": syn.get("heating_temperature"), "duration": f"{syn.get('heating_time')} min",
                                                "furnace": syn.get("furnace_name")}),
                  Action(action="cool"), Action(action="transfer", params={"to": "XRD holder"}),
                  Action(action="measure", params={"method": "powder XRD"})]
        success = None if cat is None else cat not in ("unreacted", "failed")
        r = Record(
            id=make_id("alab", s["sample_id"]), source="alab_precursor_genome", source_id=s["sample_id"], source_url=url(f),
            license=lic(f), modality="run_log", provenance="measurement", physical_scale="material",
            method="automated solid-state synthesis + XRD (A-Lab)", instrument=syn.get("furnace_name"),
            property="mass_recovered", value=syn.get("mass_recovered_mg"), units="mg",
            entity={"target": s.get("target_compound"), "precursors": precs, "elements": syn.get("elements_present")},
            conditions={"heating_temperature_C": syn.get("heating_temperature"), "heating_time_min": syn.get("heating_time"),
                        "dosing_error_g": dose_err, "met_target_mass": syn.get("met_target_mass"),
                        "temperature_profile": {"t_min": [round(x, 3) for x in tmin[:200]], "T_C": tc[:200]},
                        "reaction_energy_ev_per_atom": s.get("reaction_energy_ev_per_atom")},
            protocol_text=f"Dose {', '.join(precs)} for {s.get('target_compound')}; heat {syn.get('heating_temperature')} C for "
                          f"{syn.get('heating_time')} min in {syn.get('furnace_name')}; XRD.",
            protocol_steps=steps, raw=raw,
            outcome=Outcome(success=success, metric=f"reaction_category={cat}"),
            group_key=f"alab|{s.get('target_compound')}",
        )
        r.annotations.append(Annotation(key="note", value=(oc.get("category_notes") or "")[:400], annotator="human:alab_team", auto=False))
        if s.get("physical_failure"):
            r.annotations.append(Annotation(key="failure_reason", value=str(s["physical_failure"])[:200], annotator="human:alab_team", auto=False))
        out.append(r)
        n += 1
    print("alab ledger", n)
    return {s["sample_id"]: (s.get("outcome") or {}).get("reaction_category") for s in d["samples"]}


# ---------------------------------------------------------------- 3. A-Lab raw XRD
def alab_xrd(cat_by_id, max_scans=400):
    f = "alab_raw_scans.zip"
    z = zipfile.ZipFile(RAW / f)
    names = [n for n in z.namelist() if n.endswith(".xrdml") and "/precursors/" not in n]
    kept = 0
    for i, n in enumerate(sorted(names)):
        if kept >= max_scans:
            break
        txt = z.read(n).decode(errors="replace")
        m_s = re.search(r"<startPosition>([\d.]+)</startPosition>", txt)
        m_e = re.search(r"<endPosition>([\d.]+)</endPosition>", txt)
        m_c = re.search(r"<(?:counts|intensities)[^>]*>([^<]+)<", txt)
        if not (m_s and m_e and m_c):
            continue
        y = np.array(m_c.group(1).split(), float)
        x = np.linspace(float(m_s.group(1)), float(m_e.group(1)), len(y))
        xs, ys = downsample(x, y)
        sid = "_".join(Path(n).name.split("_")[:2])
        if kept == 0:
            (SAMPLES / Path(n).name).write_text(txt)
            raw = str(SAMPLES / Path(n).name)
        out.append(Record(
            id=make_id("alabxrd", n), source="alab_xrd_raw", source_id=Path(n).name, source_url=url(f), license=lic(f),
            modality="spectrum", provenance="measurement", physical_scale="material", method="powder XRD (Cu K-alpha)",
            instrument="Aeris (Malvern Panalytical)", property="xrd_pattern", units="counts vs 2theta_deg",
            entity={"sample_id": sid, "file": Path(n).name},
            conditions={"x": xs, "y": ys, "n_points_raw": int(len(y)), "max_counts": float(y.max())},
            raw=raw, outcome=Outcome(success=None if cat_by_id.get(sid) is None else cat_by_id[sid] not in ("unreacted", "failed"),
                                     metric=f"reaction_category={cat_by_id.get(sid)}"),
            group_key=f"alab|{sid}",
        ))
        kept += 1
    print("alab xrd", kept)


# ---------------------------------------------------------------- 4. ChemOS log
def chemos_log():
    import csv
    f = "chemos_opticslog.csv"
    shutil.copy(RAW / f, SAMPLES / f)
    for row in csv.DictReader(open(RAW / f)):
        status = " ".join((row["status"] or "").split())  # collapse multi-line status
        fail = bool(re.search(r"fail|error|abort", status, re.I))
        out.append(annotate(Record(
            id=make_id("chemos", row["id"]), source="chemos2_log", source_id=row["id"], source_url=url(f), license=lic(f),
            modality="run_log", provenance="measurement", physical_scale="lab-process", method="ChemOS 2.0 closed-loop optics log",
            entity={"job_id": row["job_id"], "device_id": row["device_id"]},
            conditions={"timestamp": row["timestamp"], "location": row["location"]}, protocol_text=status,
            raw=str(SAMPLES / f), outcome=Outcome(success=False if fail else None, metric="status"), group_key="chemos|optics",
        )))


# ---------------------------------------------------------------- 5-7. RODARE compound 10a (NMR, HPLC, IR)
def rodare():
    f = "nmr_13c_10a.zip"
    z = zipfile.ZipFile(RAW / f)
    log = z.read("NMR_13C_cmpd_10a/log").decode(errors="replace")
    procpar = z.read("NMR_13C_cmpd_10a/procpar").decode(errors="replace")
    (SAMPLES / "nmr_13c_10a.log.txt").write_text(log)
    (SAMPLES / "nmr_13c_10a.procpar.txt").write_text(procpar[:20000])
    sfrq = re.search(r"^sfrq .*\n\d+ ([\d.]+)", procpar, re.M)
    aborted = "aborted" in log.lower()
    blocks = len(re.findall(r"BS \d+ completed", log))
    r = Record(
        id=make_id("rodare", "nmr10a"), source="rodare_cmpd10a", source_id="NMR_13C_cmpd_10a", source_url=url(f), license=lic(f),
        modality="spectrum", provenance="measurement", physical_scale="molecular", method="13C NMR FID (Varian)",
        instrument="Varian NMR", property="nmr_fid_blocks_completed", value=blocks, units="blocks",
        entity={"compound": "aryl-boronic ester 10a"}, conditions={"spectrometer_freq_MHz": float(sfrq.group(1)) if sfrq else None},
        protocol_text=log.strip(), raw=str(SAMPLES / "nmr_13c_10a.log.txt"),
        outcome=Outcome(success=not aborted, metric="acquisition_completed"), group_key="rodare|10a",
    )
    if aborted:
        r.annotations.append(Annotation(key="failure_reason", value="acquisition_aborted", annotator="rule:v0"))
    out.append(r)

    f = "hplc_10a.csv"
    keep_head(f, 300)
    xy = np.loadtxt(RAW / f, delimiter=",")
    _, uniq = np.unique(xy[:, 0], return_index=True)  # drop duplicate rounded timestamps
    xs, ys = downsample(xy[uniq, 0], xy[uniq, 1])
    out.append(Record(
        id=make_id("rodare", "hplc10a"), source="rodare_cmpd10a", source_id="HPLC_cmpd_10a", source_url=url(f), license=lic(f),
        modality="spectrum", provenance="measurement", physical_scale="molecular", method="HPLC detector trace",
        property="chromatogram", units="signal vs min", entity={"compound": "aryl-boronic ester 10a"},
        conditions={"x": xs, "y": ys, "n_points_raw": int(len(xy)), "n_duplicate_t_dropped": int(len(xy) - len(uniq)),
                    "n_negative": int((xy[:, 1] < 0).sum())},
        raw=str(SAMPLES / (f + ".head.txt")), group_key="rodare|10a"))

    f = "ir_10a.csv"
    keep_head(f, 300)
    txt = (RAW / f).read_text(errors="replace").replace(",", ".")  # decimal commas -> dots
    xy = np.loadtxt(io.StringIO(txt), delimiter=";")
    xs, ys = downsample(xy[:, 0], xy[:, 1])
    out.append(Record(
        id=make_id("rodare", "ir10a"), source="rodare_cmpd10a", source_id="IR_cmpd_10a", source_url=url(f), license=lic(f),
        modality="spectrum", provenance="measurement", physical_scale="molecular", method="FT-IR transmittance",
        property="ir_spectrum", units="%T vs cm^-1", entity={"compound": "aryl-boronic ester 10a"},
        conditions={"x": xs, "y": ys, "n_points_raw": int(len(xy))}, raw=str(SAMPLES / (f + ".head.txt")), group_key="rodare|10a"))


# ---------------------------------------------------------------- 8. DaRUS HPLC workbook (212 sheets)
def darus():
    f = "darus_hplc.xlsx"
    wb = openpyxl.load_workbook(RAW / f, read_only=True, data_only=True)
    for k, name in enumerate(wb.sheetnames):
        rows = [r[:2] for r in wb[name].iter_rows(values_only=True)]
        rows = [r for r in rows if isinstance(r[0], (int, float)) and isinstance(r[1], (int, float))]
        if len(rows) < 10:
            continue
        xy = np.array(rows, float)
        if k == 0:
            (SAMPLES / f"darus_{name}.csv").write_text("\n".join(f"{a},{b}" for a, b in xy[:500]))
        xs, ys = downsample(xy[:, 0], xy[:, 1])
        neg = "negative" in name.lower()
        out.append(Record(
            id=make_id("darus", name), source="darus_hydrolase_hplc", source_id=name, source_url=url(f), license=lic(f),
            modality="spectrum", provenance="measurement", physical_scale="molecular", method="HPLC absorbance chromatogram",
            property="chromatogram", units="mAU vs min", entity={"sheet": name, "is_negative_control": neg},
            conditions={"x": xs, "y": ys, "n_points_raw": int(len(xy)), "t_start_min": float(xy[0, 0])},
            raw=str(SAMPLES / f"darus_{wb.sheetnames[0]}.csv"), group_key=f"darus|{name.split('_')[0]}"))


# ---------------------------------------------------------------- 9-10. plate readers
def _cell(v):
    if isinstance(v, datetime.time):
        return v.hour * 60 + v.minute + v.second / 60
    return v


def plate_endpoint():
    """Cytation3 export: metadata block, layout grid, then a 'Results' table (Well ID, Well, Conc/Dil, reads...)."""
    f = "platero_plate1.xlsx"
    shutil.copy(RAW / f, SAMPLES / f)
    wb = openpyxl.load_workbook(RAW / f, read_only=True, data_only=True)
    for sheet in wb.sheetnames:
        rows = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        temps = [r[1] for r in rows if r and r[0] == "Actual Temperature:" and isinstance(r[1], (int, float))]
        hi = next((i for i, r in enumerate(rows) if "Well" in r and "Well ID" in r), None)
        if hi is None:
            continue
        header = rows[hi]
        c_id, c_well, c_conc = header.index("Well ID"), header.index("Well"), header.index("Conc/Dil")
        reads = [(c, h) for c, h in enumerate(header) if isinstance(h, str) and (h.startswith("F:") or h.startswith("OD"))]
        well_id = None
        for r in rows[hi + 1:]:
            if not r[c_well]:
                continue
            well_id = r[c_id] or well_id  # Well ID is only written on the first row of each group
            vals = {h: r[c] for c, h in reads if isinstance(r[c], (int, float))}
            main = vals.get("F:G60:485,528")
            out.append(Record(
                id=make_id("platero", sheet, r[c_well]), source="platero_cytation3", source_id=f"{sheet}/{r[c_well]}",
                source_url=url(f), license=lic(f), modality="table", provenance="measurement", physical_scale="molecular",
                method="plate reader fluorescein fluorescence (Ex485/Em528) + OD600", instrument="BioTek Cytation3",
                property="fluorescence_gain60", value=float(main) if main is not None else None, units="RFU",
                entity={"plate": sheet, "well": r[c_well], "well_id": well_id, "control": "blank" if well_id == "BLK" else None},
                conditions={"conc_dil": r[c_conc], "reads": vals, "plate_temp_C_range": [min(temps), max(temps)] if temps else None},
                raw=str(SAMPLES / f), group_key=f"platero|{sheet}"))


def plate_kinetic():
    f = "growth_curves.xlsx"
    shutil.copy(RAW / f, SAMPLES / f)
    wb = openpyxl.load_workbook(RAW / f, read_only=True, data_only=True)
    rows = [list(r) for r in wb[wb.sheetnames[0]].iter_rows(values_only=True)]
    hi = next(i for i, r in enumerate(rows) if len(r) > 2 and r[1] == "Time")
    header = rows[hi]
    data = []
    for r in rows[hi + 1:]:
        if r[1] is None:
            break
        data.append([_cell(v) for v in r])
    for c, well in enumerate(header):
        if c < 3 or not well:
            continue
        t = [d[1] for d in data if isinstance(d[c], (int, float))]
        y = [d[c] for d in data if isinstance(d[c], (int, float))]
        if not y:
            continue
        ctrl = "blank" if well == "BLK" else "contamination_control" if "Contam" in str(well) else None
        out.append(Record(
            id=make_id("growth", well, c), source="growth_curves_cytation5", source_id=f"{well}#col{c}", source_url=url(f),
            license=lic(f), modality="table", provenance="measurement", physical_scale="cell",
            method="kinetic OD600 growth curve, 30C", instrument="BioTek Cytation5", property="max_OD600", value=float(max(y)),
            units="OD", entity={"strain_well_label": well, "control": ctrl},
            conditions={"t_min": t, "od600": y, "temperature_setpoint_C": 30}, raw=str(SAMPLES / f),
            outcome=Outcome(success=(max(y) - y[0]) > 0.05, metric="growth_detected (deltaOD>0.05)"),
            group_key=f"growth|{str(well).split('-')[0]}"))


# ---------------------------------------------------------------- 11. ELN html
def eln():
    f = "eln_ca_imaging.html"
    shutil.copy(RAW / f, SAMPLES / f)
    html = (RAW / f).read_text(errors="replace")
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    text = re.sub(r"<br\s*/?>|</p>|</tr>|</li>|</h\d>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    paras = [" ".join(p.split()) for p in text.split("\n")]
    paras = [p for p in paras if len(p) > 30]
    for i, p in enumerate(paras):
        out.append(annotate(Record(
            id=make_id("eln", i), source="eln_ca_imaging", source_id=f"ro-crate_1021#para{i}", source_url=url(f), license=lic(f),
            modality="lab_note", provenance="human_annotation", physical_scale="cell", method="electronic lab notebook (free text)",
            protocol_text=p[:1500], protocol_steps=parse_procedure(p), raw=str(SAMPLES / f), group_key="eln|1021")))


# ---------------------------------------------------------------- 12. microscopy tiff
def microscopy():
    import tifffile
    from PIL import Image
    f = "microscopy_fig2c.tif"
    img = tifffile.imread(RAW / f)
    img = np.asarray(img, float)
    if img.ndim == 3 and img.shape[0] > 4:
        img = np.moveaxis(img, -1, 0)
    chans = img if img.ndim == 3 else img[None]
    thumb = chans[0][::16, ::16]
    Image.fromarray((255 * (thumb - thumb.min()) / (np.ptp(thumb) + 1e-9)).astype(np.uint8)).save(SAMPLES / "microscopy_fig2c.thumb.png")
    H, W = chans.shape[1:]
    P = 256
    for i in range(0, H - P + 1, P):
        for j in range(0, W - P + 1, P):
            patch = chans[:, i:i + P, j:j + P]
            small = patch[0][::8, ::8]  # 32x32 for the image encoder
            out.append(Record(
                id=make_id("micro", i, j), source="microscopy_alemany_fig2c", source_id=f"patch_{i}_{j}", source_url=url(f), license=lic(f),
                modality="image", provenance="measurement", physical_scale="cell", method="fluorescence microscopy (2 channels)",
                property="mean_intensity_ch0", value=float(patch[0].mean()), units="a.u.",
                entity={"patch_xy": [i, j], "channels": int(chans.shape[0])},
                conditions={"pixels_32x32_ch0": [[round(float(v), 1) for v in row] for row in small],
                            "mean_per_channel": [float(c.mean()) for c in patch], "std_per_channel": [float(c.std()) for c in patch]},
                raw=str(SAMPLES / "microscopy_fig2c.thumb.png"), group_key=f"micro|{i // 1024}_{j // 1024}"))


ada_log()
cats = alab_ledger()
alab_xrd(cats)
chemos_log()
rodare()
darus()
plate_endpoint()
plate_kinetic()
eln()
microscopy()
out = [annotate(r) if r.modality in ("run_log",) and not any(a.key == "safety_level" for a in r.annotations) else r for r in out]
print("messy records:", write_jsonl(out, Path("data/interim/messy.jsonl")))
