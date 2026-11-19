"""VASP input generation for the steps relax -> scf -> wannier (-> bands)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# VASP-recommended PAW variants (subset). Anything missing: try el, el_pv, el_sv.
RECOMMENDED = {
    "Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv", "Ca": "Ca_sv",
    "Sr": "Sr_sv", "Ba": "Ba_sv", "Sc": "Sc_sv", "Ti": "Ti_sv", "V": "V_sv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe", "Co": "Co", "Ni": "Ni", "Cu": "Cu", "Zn": "Zn", "Ga": "Ga_d",
    "Ge": "Ge_d", "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv", "Tc": "Tc_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd", "Ag": "Ag", "Cd": "Cd", "In": "In_d", "Sn": "Sn_d",
    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_sv", "Re": "Re", "Os": "Os", "Ir": "Ir", "Pt": "Pt",
    "Au": "Au", "Tl": "Tl_d", "Pb": "Pb_d", "Bi": "Bi_d", "Gd": "Gd", "Th": "Th",
}
MAGMOM0 = {"Cr": 3.0, "Mn": 4.0, "Fe": 3.0, "Co": 2.0, "Ni": 1.0, "V": 2.0, "Gd": 7.0, "Eu": 7.0, "Ru": 1.0}
F_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th", "U"}


def elements_in_order(struct: Structure) -> list[str]:
    seen = []
    for s in struct:
        el = str(s.specie)
        if el not in seen:
            seen.append(el)
    return seen


def write_potcar(struct: Structure, potcar_dir: str, dest: Path) -> dict:
    """Concatenate POTCARs. Returns {"ok", "files", "encut_suggest"}. Without a library
    (mock / laptop), writes POTCAR.spec only."""
    files, enmax = [], []
    base = Path(potcar_dir)
    for el in elements_in_order(struct):
        cands = [RECOMMENDED.get(el, el), el, f"{el}_pv", f"{el}_sv"]
        hit = next((base / c / "POTCAR" for c in cands if (base / c / "POTCAR").exists()), None)
        if hit is None:
            files.append(None)
            continue
        files.append(hit)
        m = re.search(r"ENMAX\s*=\s*([\d.]+)", hit.read_text(errors="ignore"))
        if m:
            enmax.append(float(m.group(1)))
    (dest / "POTCAR.spec").write_text("\n".join(
        f"{el} {f.parent.name if f else RECOMMENDED.get(el, el) + ' (NOT FOUND)'}"
        for el, f in zip(elements_in_order(struct), files)) + "\n")
    if all(files):
        with open(dest / "POTCAR", "w") as out:
            for f in files:
                out.write(f.read_text())
        return {"ok": True, "encut_suggest": round(1.3 * max(enmax)) if enmax else 520}
    return {"ok": False, "encut_suggest": 520,
            "missing": [el for el, f in zip(elements_in_order(struct), files) if f is None]}


def base_incar(struct: Structure, spin: str, encut: float) -> dict:
    inc = {
        "SYSTEM": struct.composition.reduced_formula, "PREC": "Accurate", "ENCUT": encut,
        "EDIFF": 1e-7, "ISMEAR": 0, "SIGMA": 0.05, "LREAL": ".FALSE.", "LASPH": ".TRUE.",
        "NELM": 200, "ALGO": "Normal", "LORBIT": 11, "NCORE": 4,
    }
    has_f = any(str(s.specie) in F_ELEMENTS for s in struct)
    has_d = any(s.specie.is_transition_metal for s in struct)
    inc["LMAXMIX"] = 6 if has_f else 4 if has_d else 2
    if spin == "collinear":
        inc["ISPIN"] = 2
        inc["MAGMOM"] = " ".join(str(MAGMOM0.get(str(s.specie), 0.0)) for s in struct)
    elif spin == "soc":
        inc.update({"LSORBIT": ".TRUE.", "LNONCOLLINEAR": ".TRUE.", "GGA_COMPAT": ".FALSE.",
                    "ISYM": -1, "SAXIS": "0 0 1"})
        inc["MAGMOM"] = " ".join(f"0 0 {MAGMOM0.get(str(s.specie), 0.0)}" for s in struct)
        inc.pop("NCORE")               # NCORE often unsafe with ncl on some builds; let VASP choose
    return inc


def write_incar(path: Path, tags: dict) -> None:
    lines = []
    for k, v in tags.items():
        if k == "WANNIER90_WIN":
            lines.append(f'WANNIER90_WIN = "\n{v.rstrip()}\n"')
        else:
            lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


def write_kpoints_grid(path: Path, grid: list[int]) -> None:
    path.write_text(f"Gamma-centred\n0\nGamma\n{grid[0]} {grid[1]} {grid[2]}\n0 0 0\n")


def write_kpoints_line(path: Path, kpath, npts=40) -> None:
    lines = ["Line-mode for Wannier comparison", str(npts), "Line-mode", "Reciprocal"]
    for a, fa, b, fb in kpath:
        lines.append(f"{fa[0]:.6f} {fa[1]:.6f} {fa[2]:.6f} ! {a}")
        lines.append(f"{fb[0]:.6f} {fb[1]:.6f} {fb[2]:.6f} ! {b}")
        lines.append("")
    path.write_text("\n".join(lines))


def prepare_vasp_step(step: str, struct: Structure, workdir: Path, cfg, spin: str,
                      kgrid_scf, kgrid_wan, kpath=None, win_text: str | None = None,
                      nbands: int | None = None, overrides: dict | None = None,
                      source_dirs: dict | None = None) -> dict:
    """Write POSCAR/INCAR/KPOINTS/POTCAR for one step. Returns a summary dict."""
    d = workdir / step
    d.mkdir(parents=True, exist_ok=True)
    Poscar(struct).write_file(d / "POSCAR")
    pot = write_potcar(struct, cfg.potcar_dir, d)
    inc = base_incar(struct, spin, pot["encut_suggest"])
    notes = []

    if step == "relax":
        inc.update({"IBRION": 2, "ISIF": 3, "NSW": 100, "EDIFFG": -0.01, "EDIFF": 1e-6,
                    "LCHARG": ".FALSE.", "LWAVE": ".FALSE."})
        write_kpoints_grid(d / "KPOINTS", kgrid_scf)
    elif step == "scf":
        inc.update({"LCHARG": ".TRUE.", "LWAVE": ".FALSE."})
        write_kpoints_grid(d / "KPOINTS", kgrid_scf)
    elif step in ("wannier", "bands"):
        chg = Path(source_dirs["scf"]) / "CHGCAR"
        if chg.exists():
            shutil.copy(chg, d / "CHGCAR")
        else:
            notes.append("CHGCAR from scf not found yet; job script will copy it at run time")
        inc.update({"ICHARG": 11, "LCHARG": ".FALSE.", "LWAVE": ".FALSE.", "ALGO": "Normal"})
        if nbands:
            inc["NBANDS"] = nbands
        if step == "wannier":
            inc.update({"ISYM": -1, "LWANNIER90": ".TRUE.", "LWRITE_MMN_AMN": ".TRUE.",
                        "LWANNIER90_RUN": ".FALSE.", "NCORE": 1, "KPAR": 1})  # safe for the W90 interface
            if spin == "soc":
                inc.pop("NCORE")   # wannier90.x runs as its own step
            if cfg.vasp_version >= 6:
                inc["WANNIER90_WIN"] = win_text
            else:
                (d / "wannier90.win").write_text(win_text)
                notes.append("VASP 5: wrote wannier90.win; check VASP appended cell and k-points")
            write_kpoints_grid(d / "KPOINTS", kgrid_wan)
        else:
            write_kpoints_line(d / "KPOINTS", kpath)
    else:
        raise ValueError(f"unknown step {step}")

    if overrides:
        for k, v in overrides.items():
            if v is None:
                inc.pop(k, None)
            else:
                inc[k] = v
    write_incar(d / "INCAR", inc)
    if not pot["ok"]:
        notes.append(f"POTCAR missing for {pot.get('missing')}; set potcar_dir in config")
    return {"step": step, "dir": str(d), "incar": {k: v for k, v in inc.items() if k != "WANNIER90_WIN"},
            "potcar_ok": pot["ok"], "notes": notes}
