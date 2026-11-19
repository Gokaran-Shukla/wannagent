"""Structure loading, description, k-point grids and high-symmetry paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

MAGNETIC = {"Cr", "Mn", "Fe", "Co", "Ni", "V", "Gd", "Eu", "Nd", "Sm", "Tb", "Dy", "Ho", "Er", "Ru"}


def load_structure(path: str | Path) -> Structure:
    return Structure.from_file(str(path))


def describe(struct: Structure) -> dict:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-3)
        sg = f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()})"
    except Exception:  # pragma: no cover - very distorted cells
        sg = "unknown"
    lat = struct.lattice
    vac_axes = [i for i, L in enumerate(lat.abc) if _has_vacuum(struct, i)]
    return {
        "formula": struct.composition.reduced_formula,
        "n_atoms": len(struct),
        "species": {str(el): int(n) for el, n in struct.composition.items()},
        "spacegroup": sg,
        "abc_A": [round(x, 4) for x in lat.abc],
        "angles_deg": [round(x, 3) for x in lat.angles],
        "volume_A3": round(lat.volume, 3),
        "possibly_magnetic": sorted({str(s.specie) for s in struct if str(s.specie) in MAGNETIC}),
        "vacuum_axes": vac_axes,          # slab/2D detection -> 1 k-point along those axes
    }


def _has_vacuum(struct: Structure, axis: int, min_gap=6.0) -> bool:
    frac = np.sort(struct.frac_coords[:, axis] % 1.0)
    gaps = np.diff(np.concatenate([frac, [frac[0] + 1.0]]))
    return gaps.max() * struct.lattice.abc[axis] > min_gap


def kgrid(struct: Structure, kppa: int, vacuum_axes=()) -> list[int]:
    """Gamma-centred grid from k-points-per-reciprocal-atom; 1 along vacuum."""
    k = Kpoints.automatic_density(struct, kppa, force_gamma=True).kpts[0]
    k = [max(1, int(x)) for x in k]
    for ax in vacuum_axes:
        k[ax] = 1
    return k


def kpath_for_wannier(struct: Structure, vacuum_axes=()) -> list[tuple[str, list[float], str, list[float]]]:
    """High-symmetry segments in fractional coordinates of the INPUT cell.

    Uses the Setyawan-Curtarolo path from pymatgen, transformed back to the given
    lattice so it can go straight into wannier90.win (kpoint_path is in the
    reciprocal basis of unit_cell_cart).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = HighSymmKpath(struct, path_type="setyawan_curtarolo", symprec=1e-3)
    labels = kp.kpath["kpoints"]
    # pymatgen's standard primitive cell may be ROTATED and re-chosen relative to the
    # input. Find the integer matrix N with A_input = N A_prim (rows = lattice vectors,
    # orientation-independent); then k_input = N k_prim (k.r phase invariance).
    mapping = kp.prim.lattice.find_mapping(struct.lattice, ltol=1e-3, atol=0.1)
    if mapping is None:
        raise RuntimeError("could not map standard primitive cell onto the input cell; "
                           "give kpoint_path manually")
    N = np.array(mapping[2], float)

    def to_input(frac_prim):
        return [round(float(x), 6) for x in N @ np.asarray(frac_prim, float)]

    segs = []
    for branch in kp.kpath["path"]:
        for a, b in zip(branch[:-1], branch[1:]):
            fa, fb = to_input(labels[a]), to_input(labels[b])
            if vacuum_axes and any(abs(fa[ax]) > 1e-6 or abs(fb[ax]) > 1e-6 for ax in vacuum_axes):
                continue                       # in-plane path only for slabs
            name = lambda s: "G" if s == "\\Gamma" else s.replace("\\", "").replace("_", "")
            segs.append((name(a), fa, name(b), fb))
    return segs
