"""Mock cluster: fakes VASP and Wannier90 outputs so the whole agent loop can be
tested on a laptop with no VASP licence and no SLURM.

The fake "DFT" is a random tight-binding model (the true Wannier manifold) plus
two bands of other orbital character: a deep one and an entangled one that
starts EF + 1.5 eV. Disentanglement is good only if dis_froz_max stays below
that entangled band, exactly the decision the agent must get right.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

EF = 5.0
ENTANGLED_BOTTOM = EF + 1.5


class MockCluster:
    def __init__(self, project):
        self.p = project
        self.calls = {}
        self.jobs = {}
        self.n = 1000

    # same interface as SlurmCluster
    def submit(self, script: Path, kind: str, spin: str) -> str:
        self.n += 1
        jid = str(self.n)
        d = script.parent
        fail = kind in self.p.cfg.mock_failures and self.p.step(kind)["attempts"] == 0
        getattr(self, f"_{kind}")(d, spin, fail)
        self.jobs[jid] = "FAILED" if fail and kind != "wannier90" else "COMPLETED"
        self.calls[jid] = 0
        return jid

    def status(self, jobid: str) -> str:
        self.calls[jobid] = self.calls.get(jobid, 0) + 1
        return "RUNNING" if self.calls[jobid] == 1 else self.jobs.get(jobid, "UNKNOWN")

    def cancel(self, jobid):
        return "CANCELLED"

    # ---------------------------------------------------------------- VASP
    def _outcar(self, d, nbands, extra=""):
        (d / "OUTCAR").write_text(
            f" NELECT =      16.0000\n NBANDS=    {nbands}\n NELM   =    200\n{extra}"
            f" E-fermi :   {EF:.4f}     XC(G=0):  -10.0\n"
            "  free  energy   TOTEN  =       -10.84320000 eV\n"
            " number of electron      16.0000000 magnetization       0.0000000\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text("DAV:   1  -0.1E+02\nDAV:   2  -0.1E+02\n   1 F= -.10843200E+02 E0= -.1E+02\n")
        (d / "vasp.out").write_text("running on 40 total cores\n")

    def _relax(self, d, spin, fail):
        self._outcar(d, 32, " reached required accuracy - stopping structural energy minimisation\n")
        (d / "CONTCAR").write_text((d / "POSCAR").read_text())

    def _scf(self, d, spin, fail):
        if fail:
            (d / "vasp.out").write_text("Error EDDDAV: Call to ZHEGV failed. Returncode = 7 2 16\n")
            (d / "OUTCAR").write_text(" NBANDS=    32\n")
            return
        self._outcar(d, 32)
        (d / "CHGCAR").write_text("mock charge density\n")

    def _bands(self, d, spin, fail):
        self._outcar(d, 32)

    def _model(self, nw):
        rng = np.random.default_rng(7)
        Rs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        H = np.zeros((len(Rs), nw, nw), complex)
        idx = {tuple(r): n for n, r in enumerate(Rs)}
        for n, R in enumerate(Rs):
            m = idx[tuple(-R)]
            if n > m:
                continue
            if not R.any():
                A = rng.normal(0, 0.3, (nw, nw)) + 1j * rng.normal(0, 0.1, (nw, nw))
                H[n] = 0.5 * (A + A.conj().T) + np.diag(np.linspace(EF - 2.5, EF + 1.0, nw))
            else:
                A = 0.25 * (rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))) / np.sqrt(nw)
                H[n], H[m] = A, A.conj().T
        return Rs, H

    def _eigs(self, Rs, H, k):
        Hk = np.einsum("kr,rmn->kmn", np.exp(2j * np.pi * k @ Rs.T), H)
        return np.linalg.eigvalsh(0.5 * (Hk + Hk.conj().transpose(0, 2, 1)))

    def _wannier(self, d, spin, fail):
        st = self.p.state
        nw, nb = st["num_wann"], st["nbands"]
        grid = [int(x) for x in (d / "KPOINTS").read_text().splitlines()[3].split()]
        k = np.array([[i / grid[0], j / grid[1], l / grid[2]] for i in range(grid[0])
                      for j in range(grid[1]) for l in range(grid[2])])
        Rs, H = self._model(nw)
        e_w = self._eigs(Rs, H, k)
        # shift the Wannier manifold to sit below the entangled band: a clear frozen-window edge
        shift = max(0.0, e_w.max() - (ENTANGLED_BOTTOM - 0.3))
        H[np.where(~Rs.any(1))[0][0]] -= shift * np.eye(nw)
        e_w = self._eigs(Rs, H, k)
        nk = len(k)
        deep = np.full((nk, 1), EF - 14.0)
        ent = ENTANGLED_BOTTOM + 2.0 * (1 - np.cos(2 * np.pi * k[:, :1]))
        high = EF + 9 + np.arange(nb - nw - 2)[None, :] + 0 * k[:, :1]
        e = np.sort(np.hstack([deep, e_w, ent, high]), axis=1)
        p = np.where((e > EF - 10) & (e < ENTANGLED_BOTTOM - 0.01), 0.95, 0.03)
        p[np.isclose(e, ent)] = 0.10

        incar = (d / "INCAR").read_text()
        win_user = re.search(r'WANNIER90_WIN = "\n(.*?)\n"', incar, re.S).group(1)
        prefixes = ["wannier90.up", "wannier90.dn"] if spin == "collinear" else ["wannier90"]
        for pre in prefixes:
            with open(d / f"{pre}.eig", "w") as f:
                for ik in range(nk):
                    for ib in range(nb):
                        f.write(f"{ib+1:5d}{ik+1:5d}{e[ik, ib]:18.12f}\n")
            with open(d / f"{pre}.amn", "w") as f:
                f.write("mock amn\n")
                f.write(f"{nb} {nk} {nw}\n")
                for ik in range(nk):
                    for n in range(nw):
                        for m in range(nb):
                            a = np.sqrt(p[ik, m] / nw)
                            f.write(f"{m+1} {n+1} {ik+1} {a:.10f} 0.0\n")
            (d / f"{pre}.mmn").write_text("mock mmn\n")
            kp = "\n".join(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in k)
            (d / f"{pre}.win").write_text(
                f"{win_user}\nnum_bands = {nb}\nmp_grid = {grid[0]} {grid[1]} {grid[2]}\n"
                f"begin kpoints\n{kp}\nend kpoints\n")
        np.save(d / ".mock_model.npy", {"Rs": Rs, "H": H}, allow_pickle=True)
        self._outcar(d, nb)

    # ------------------------------------------------------------- Wannier90
    def _wannier90(self, d, spin, fail):
        mdl = np.load(d / ".mock_model.npy", allow_pickle=True).item()
        Rs, H = mdl["Rs"], mdl["H"].copy()
        nw = H.shape[1]
        for pre in (["wannier90.up", "wannier90.dn"] if spin == "collinear" else ["wannier90"]):
            win = (d / f"{pre}.win").read_text()
            fmax = float(re.search(r"dis_froz_max\s*=\s*(\S+)", win).group(1))
            bad = max(0.0, fmax - (ENTANGLED_BOTTOM - 0.05))
            Hw = H.copy()
            if bad > 0:          # entangled band forced into the frozen window: model distorts
                Hw[np.where(~Rs.any(1))[0][0]] += np.diag(np.linspace(0, 0.3 + bad, nw))
            spread = 1.8 + 25 * bad
            with open(d / f"{pre}_hr.dat", "w") as f:
                f.write("written by mock wannier90\n")
                f.write(f"{nw}\n{len(Rs)}\n")
                deg = ["1"] * len(Rs)
                for i in range(0, len(deg), 15):
                    f.write(" ".join(deg[i:i + 15]) + "\n")
                for r, R in enumerate(Rs):
                    for n in range(nw):
                        for m in range(nw):
                            v = Hw[r, m, n]
                            f.write(f"{R[0]:5d}{R[1]:5d}{R[2]:5d}{m+1:5d}{n+1:5d}{v.real:12.6f}{v.imag:12.6f}\n")
            wf = "\n".join(f"  WF centre and spread {i+1:4d}  (  0.000000,  0.000000,  0.000000 )"
                           f"  {spread + 0.1 * (i % 3):10.6f}" for i in range(nw))
            (d / f"{pre}.wout").write_text(
                "  DISENTANGLE\n  <<<     Disentanglement convergence criteria satisfied     >>>\n"
                "  <<<     Wannierisation convergence criteria satisfied     >>>\n"
                f" Final State\n{wf}\n  Omega I      =  {nw*1.2:.6f}\n  Omega D      =  0.1\n"
                f"  Omega OD     =  {nw*0.6:.6f}\n  Omega Total  =  {nw*spread:.6f}\n All done: wannier90 exiting\n")
