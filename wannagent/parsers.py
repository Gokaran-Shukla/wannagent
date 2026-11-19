"""Parsers for VASP and Wannier90 outputs, and the numerical quality checks."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Known VASP failure messages -> suggested INCAR fix (the agent reads these)
VASP_ERRORS = {
    "ZBRENT": "relaxation line search failed: use IBRION=1 or restart from CONTCAR with smaller POTIM",
    "EDDDAV": "Davidson failed: set ALGO=Normal or ALGO=All, or reduce NCORE",
    "Sub-Space-Matrix is not hermitian": "set ALGO=Fast->Normal, or ALGO=All with smaller TIME",
    "BRMIX: very serious problems": "charge mixing problem: lower AMIX/BMIX or set IMIX=1",
    "VERY BAD NEWS": "internal error: often NBANDS/NPAR mismatch; set NCORE=1 and rerun",
    "Error reading item 'MAGMOM'": "MAGMOM length wrong: 1 value per atom (collinear) or 3 (ncl)",
    "POTCAR": "POTCAR problem: check potcar_dir and element order",
    "num_bands": "Wannier90 num_bands/num_wann inconsistency: raise NBANDS",
    "LAPACK: Routine ZPOTRF failed": "Cholesky failure: set ALGO=Normal and check POSCAR overlaps",
    "too few bands": "increase NBANDS",
}


def parse_vasp_dir(d: Path) -> dict:
    d = Path(d)
    out = {"dir": str(d), "finished": False, "errors": []}
    outcar = d / "OUTCAR"
    log = (d / "vasp.out").read_text(errors="ignore") if (d / "vasp.out").exists() else ""
    if outcar.exists():
        t = outcar.read_text(errors="ignore")
        out["finished"] = "General timing and accounting" in t
        f = re.findall(r"E-fermi\s*:\s*(-?[\d.]+)", t)
        out["efermi_eV"] = float(f[-1]) if f else None
        e = re.findall(r"free  energy   TOTEN\s*=\s*(-?[\d.]+)", t)
        out["toten_eV"] = float(e[-1]) if e else None
        nb = re.search(r"NBANDS=\s*(\d+)", t)
        out["nbands"] = int(nb.group(1)) if nb else None
        ne = re.search(r"NELECT\s*=\s*([\d.]+)", t)
        out["nelect"] = float(ne.group(1)) if ne else None
        mag = re.findall(r"number of electron\s+[\d.]+\s+magnetization\s+(-?[\d.]+)", t)
        out["total_magnetization"] = float(mag[-1]) if mag else None
        out["relax_converged"] = "reached required accuracy" in t
        nelm = re.search(r"NELM\s*=\s*(\d+)", t)
        out["nelm"] = int(nelm.group(1)) if nelm else None
    osz = d / "OSZICAR"
    if osz.exists():
        blocks = re.split(r"^\s*\d+ F=", osz.read_text(), flags=re.M)
        last = blocks[-2] if len(blocks) > 1 else blocks[-1]
        n_el = len(re.findall(r"^(DAV|RMM|CG|SDA):", last, re.M))
        out["electronic_steps_last_ionic"] = n_el
        out["scf_converged"] = out.get("nelm") is None or n_el < out["nelm"]
    for key, fix in VASP_ERRORS.items():
        if key in log:
            out["errors"].append({"match": key, "suggested_fix": fix})
    if not out["finished"] and not out["errors"] and log:
        out["log_tail"] = log.strip().splitlines()[-15:]
    return out


def parse_wout(path: Path) -> dict:
    t = Path(path).read_text(errors="ignore")
    res = {"file": str(path)}
    res["disentanglement_converged"] = ("Disentanglement convergence criteria satisfied" in t
                                        or "DISENTANGLE" not in t)
    res["wannierisation_converged"] = "Wannierisation convergence criteria satisfied" in t
    final = t.split("Final State")[-1] if "Final State" in t else t
    spreads = [float(x) for x in re.findall(
        r"WF centre and spread\s+\d+\s+\(\s*[-\d.]+,\s*[-\d.]+,\s*[-\d.]+\s*\)\s+([\d.]+)", final)]
    centres = re.findall(
        r"WF centre and spread\s+\d+\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)", final)
    res["spreads_A2"] = spreads
    res["max_spread_A2"] = max(spreads) if spreads else None
    res["mean_spread_A2"] = float(np.mean(spreads)) if spreads else None
    res["centres_A"] = [[float(c) for c in xyz] for xyz in centres]
    for name, key in (("Omega I", "omega_I"), ("Omega D", "omega_D"), ("Omega OD", "omega_OD"),
                      ("Omega Total", "omega_total")):
        m = re.findall(rf"{name}\s*=\s*([\d.]+)", final)
        res[key] = float(m[-1]) if m else None
    errs = re.findall(r"(Error:.*|ERROR.*|Exiting.*)", t)
    werr = Path(path).with_suffix(".werr")
    if werr.exists():
        errs.append(werr.read_text().strip()[:500])
    res["errors"] = errs
    res["finished"] = "All done" in t
    return res


def read_eig(path: Path) -> np.ndarray:
    a = np.loadtxt(path)
    nb, nk = int(a[:, 0].max()), int(a[:, 1].max())
    e = np.zeros((nk, nb))
    e[a[:, 1].astype(int) - 1, a[:, 0].astype(int) - 1] = a[:, 2]
    return e


def read_amn_projectability(path: Path) -> np.ndarray:
    """p[k, m] = sum_n |A_mn(k)|^2: how much band m is spanned by the trial orbitals."""
    with open(path) as f:
        f.readline()
        nb, nk, nw = map(int, f.readline().split())
    a = np.loadtxt(path, skiprows=2)
    m, k = a[:, 0].astype(int) - 1, a[:, 2].astype(int) - 1
    p = np.zeros((nk, nb))
    np.add.at(p, (k, m), a[:, 3] ** 2 + a[:, 4] ** 2)
    return np.clip(p, 0, 1)


def read_win_kpoints(path: Path) -> np.ndarray:
    m = re.search(r"begin\s+kpoints(.*?)end\s+kpoints", Path(path).read_text(), re.S | re.I)
    if not m:
        raise ValueError("no kpoints block in .win (VASP writes it during the wannier step)")
    rows = [l.split()[:3] for l in m.group(1).strip().splitlines() if l.strip()]
    return np.array(rows, float)


def read_hr(path: Path):
    lines = Path(path).read_text().splitlines()
    nw, nr = int(lines[1]), int(lines[2])
    ndeg_lines = (nr + 14) // 15
    deg = np.array(" ".join(lines[3:3 + ndeg_lines]).split(), int)
    data = np.array(" ".join(lines[3 + ndeg_lines:]).split(), float).reshape(-1, 7)
    R = data[::nw * nw, :3].astype(int)
    H = (data[:, 5] + 1j * data[:, 6]).reshape(nr, nw, nw).transpose(0, 2, 1)  # file order: m fastest
    return R, deg, H


def wannier_eigs(R, deg, H, kfrac: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * kfrac @ R.T) / deg[None, :]       # (nk, nR)
    Hk = np.einsum("kr,rmn->kmn", phase, H)
    Hk = 0.5 * (Hk + Hk.conj().transpose(0, 2, 1))
    return np.linalg.eigvalsh(Hk)


def _window_errors(e_dft, e_wan, lo, hi):
    errs = []
    for ed, ew in zip(e_dft, e_wan):
        for e in ed[(ed >= lo) & (ed <= hi)]:
            errs.append(np.min(np.abs(ew - e)))
    errs = np.array(errs) * 1000
    return {"n_states_compared": int(errs.size),
            "max_err_meV": float(errs.max()) if errs.size else None,
            "rms_err_meV": float(np.sqrt((errs ** 2).mean())) if errs.size else None}


def frozen_window_check(wdir: Path, prefix: str, froz_min: float, froz_max: float) -> dict:
    """Wannier H(R) evaluated at the DFT k-grid must reproduce DFT eigenvalues
    inside the frozen window (to ~1 meV). Larger errors = bad disentanglement."""
    wdir = Path(wdir)
    k = read_win_kpoints(wdir / f"{prefix}.win")
    e_dft = read_eig(wdir / f"{prefix}.eig")
    R, deg, H = read_hr(wdir / f"{prefix}_hr.dat")
    return {"prefix": prefix, "window_eV": [froz_min, froz_max],
            **_window_errors(e_dft, wannier_eigs(R, deg, H, k), froz_min, froz_max)}


def bands_vs_vasp(wdir: Path, bands_dir: Path, prefix: str, froz_min: float, froz_max: float) -> dict:
    """Off-grid test along the high-symmetry path: VASP line-mode vs Wannier interpolation."""
    from pymatgen.io.vasp import Vasprun
    from pymatgen.electronic_structure.core import Spin
    vr = Vasprun(str(Path(bands_dir) / "vasprun.xml"), parse_projected_eigen=False, parse_potcar_file=False)
    k = np.array(vr.actual_kpoints)
    spin = Spin.down if prefix.endswith(".dn") else Spin.up
    e_dft = vr.eigenvalues[spin][:, :, 0]
    R, deg, H = read_hr(Path(wdir) / f"{prefix}_hr.dat")
    return {"prefix": prefix, "path_points": len(k),
            **_window_errors(e_dft, wannier_eigs(R, deg, H, k), froz_min, froz_max)}


def suggest_windows(wdir: Path, prefix: str, num_wann: int, efermi: float) -> dict:
    """Projectability-based windows (cf. Qiao et al., npj Comput. Mater. 9, 208 (2023)).

    - outer window: all bands with projectability > 0.05
    - frozen window: highest energy below which every band is well projected (p > 0.85)
      and no k-point has more than num_wann bands inside it
    """
    wdir = Path(wdir)
    e = read_eig(wdir / f"{prefix}.eig")
    p = read_amn_projectability(wdir / f"{prefix}.amn")
    good = p > 0.25
    if not good.any():
        return {"error": "no band has projectability > 0.25: projections are wrong for this material"}
    win_min = float(e[good].min() - 0.2)
    win_max = float(e[p > 0.05].max() + 0.5)
    lo = win_min
    cands = np.sort(np.unique(np.round(e[(e >= lo) & (e < win_max)], 3)))

    def ok(E):          # monotonic: once False for some E, False for all larger E
        inside = (e >= lo) & (e <= E)
        return inside.sum(1).max() <= num_wann and not (p[inside] < 0.85).any()

    i_lo, i_hi = -1, len(cands)          # invariant: ok(cands[i_lo]) True, ok(cands[i_hi]) False
    while i_hi - i_lo > 1:
        mid = (i_lo + i_hi) // 2
        if ok(cands[mid]):
            i_lo = mid
        else:
            i_hi = mid
    if i_lo < 0:
        return {"error": "even the lowest projected band violates the frozen-window rules; "
                         "check projections or set dis_froz_max manually",
                "dis_win_min": round(win_min, 4), "dis_win_max": round(win_max, 4)}
    froz_max = float(cands[i_lo])
    # stop just below the next eigenvalue, or 50 meV under this one if they are close
    nxt = float(cands[i_hi]) if i_hi < len(cands) else froz_max + 0.1
    froz_max = min(froz_max + 0.5 * (nxt - froz_max), froz_max + 0.05)
    n_froz = int(((e >= lo) & (e <= froz_max)).sum(1).max())
    return {"dis_win_min": round(win_min, 4), "dis_froz_min": round(win_min, 4),
            "dis_froz_max": round(froz_max, 4), "dis_win_max": round(win_max, 4),
            "froz_max_minus_EF_eV": round(froz_max - efermi, 3),
            "max_bands_in_frozen_window": n_froz, "num_wann": num_wann,
            "mean_projectability_occupied": float(p[e < efermi].mean()) if (e < efermi).any() else None}


def find_prefixes(wdir: Path) -> list[str]:
    wdir = Path(wdir)
    if (wdir / "wannier90.up.win").exists():
        return ["wannier90.up", "wannier90.dn"]
    return ["wannier90"]
