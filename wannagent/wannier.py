"""Wannier90 setup: projections, num_wann, energy windows, .win rendering and editing."""
from __future__ import annotations

import re
from pathlib import Path

from pymatgen.core import Element, Structure

NORB = {"s": 1, "p": 3, "d": 5, "f": 7, "sp": 2, "sp2": 3, "sp3": 4, "sp3d2": 6,
        "px": 1, "py": 1, "pz": 1, "dxy": 1, "dxz": 1, "dyz": 1, "dz2": 1, "dx2-y2": 1,
        "t2g": 3, "eg": 2}
SP3_HOSTS = {"C", "Si", "Ge"}          # covalent tetrahedral semiconductors


def default_orbitals(el: str) -> list[str]:
    e = Element(el)
    if el == "H":
        return ["s"]
    if e.is_transition_metal or el in {"La", "Lu"}:
        return ["d"]
    if e.is_lanthanoid or e.is_actinoid:
        return ["d"]                    # f is usually localised -> treat as core/DFT+U; agent warns
    if el in SP3_HOSTS:
        return ["sp3"]
    if e.block == "p":
        return ["p"]
    return []                           # alkali / alkaline earth: ionic, no projections


def suggest_projections(struct: Structure) -> dict[str, list[str]]:
    out = {}
    for el in sorted({str(s.specie) for s in struct}):
        orbs = default_orbitals(el)
        if orbs:
            out[el] = orbs
    return out


def count_wann(struct: Structure, projections: dict[str, list[str]], soc: bool) -> int:
    n = 0
    for site in struct:
        for orb in projections.get(str(site.specie), []):
            for o in orb.split(";"):
                n += NORB[o.strip()]
    return n * (2 if soc else 1)


def zval_from_potcar(potcar: Path) -> dict[str, float]:
    """Element -> valence electrons from an assembled POTCAR (order matters)."""
    if not potcar.exists():
        return {}
    txt = potcar.read_text(errors="ignore")
    titles = re.findall(r"TITEL\s*=\s*PAW_PBE\s+(\S+)", txt)
    zvals = [float(z) for z in re.findall(r"ZVAL\s*=\s*([\d.]+)", txt)]
    return {t.split("_")[0]: z for t, z in zip(titles, zvals)}


def suggest_nbands(struct: Structure, num_wann: int, soc: bool, zval: dict[str, float]) -> int:
    nelect = sum(zval.get(str(s.specie), 8.0) for s in struct)
    occ = nelect if soc else nelect / 2
    nb = int(max(1.3 * occ, occ + num_wann)) + 10
    return nb + (nb % 2 if soc else 0)


def render_win(params: dict) -> str:
    """Only physics keys. VASP 6 adds unit cell, atoms, mp_grid and k-points itself."""
    lines = [f"num_wann = {params['num_wann']}"]
    for k in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
        if params.get(k) is not None:
            lines.append(f"{k} = {params[k]:.4f}")
    lines += [
        f"dis_num_iter = {params.get('dis_num_iter', 2000)}",
        f"num_iter = {params.get('num_iter', 1000)}",
        "dis_conv_tol = 1.0e-10",
        "conv_tol = 1.0e-10",
        "guiding_centres = true",
        "write_hr = true",
        "write_xyz = true",
        "write_tb = true",
        "use_ws_distance = true",
    ]
    if params.get("soc"):
        lines.append("spinors = true")
    lines.append("begin projections")
    if params.get("random_projections"):
        lines.append("random")
    for el, orbs in params["projections"].items():
        lines.append(f"{el}:{';'.join(orbs)}")
    lines.append("end projections")
    if params.get("kpath"):
        lines += ["bands_plot = true", "bands_num_points = 100", "begin kpoint_path"]
        for a, fa, b, fb in params["kpath"]:
            lines.append(f"{a} {fa[0]:.6f} {fa[1]:.6f} {fa[2]:.6f}  {b} {fb[0]:.6f} {fb[1]:.6f} {fb[2]:.6f}")
        lines.append("end kpoint_path")
    return "\n".join(lines) + "\n"


_BLOCK = re.compile(r"begin\s+(\w+)(.*?)end\s+\1", re.S | re.I)


def update_win_file(path: Path, changes: dict) -> dict:
    """Edit key = value lines (and the projections block) of an existing .win in place.

    Geometry and k-point blocks that VASP wrote are left untouched, so a rerun of
    wannier90.x with new windows needs no new VASP run.
    Returns {key: (old, new)}.
    """
    txt = path.read_text()
    log = {}
    for key, val in changes.items():
        if key == "projections":
            body = "\n".join(f"{el}:{';'.join(o)}" for el, o in val.items())
            new_block = f"begin projections\n{body}\nend projections"
            m = re.search(r"begin\s+projections.*?end\s+projections", txt, re.S | re.I)
            old = m.group(0) if m else None
            txt = txt.replace(old, new_block) if m else txt + "\n" + new_block + "\n"
            log[key] = (old, new_block)
            continue
        sval = f"{val:.4f}" if isinstance(val, float) else str(val).lower() if isinstance(val, bool) else str(val)
        pat = re.compile(rf"^(\s*{key}\s*[=:]\s*)(\S+)(.*)$", re.M | re.I)
        m = pat.search(txt)
        if m:
            log[key] = (m.group(2), sval)
            txt = pat.sub(lambda mm: f"{mm.group(1)}{sval}{mm.group(3)}", txt, count=1)
        else:
            log[key] = (None, sval)
            txt = f"{key} = {sval}\n" + txt
    path.write_text(txt)
    return log


def read_win_value(path: Path, key: str) -> str | None:
    m = re.search(rf"^\s*{key}\s*[=:]\s*(\S+)", path.read_text(), re.M | re.I)
    return m.group(1) if m else None
