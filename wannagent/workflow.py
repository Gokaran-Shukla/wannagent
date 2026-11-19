"""Project state and the actions an agent (or the fixed pipeline) can take.

Every action returns a JSON-serialisable dict, so the same functions serve as
LLM tools. State lives in <workdir>/wannagent_state.json and survives restarts:
the agent can be stopped while VASP runs and resumed hours later.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from pymatgen.core import Structure

from . import parsers, structure as st, wannier as wn
from .config import Config
from .slurm import SlurmCluster, job_script
from .vasp_inputs import prepare_vasp_step

ORDER = ["relax", "scf", "wannier", "wannier90", "bands"]
DONE_STATES = {"COMPLETED"}
FAIL_STATES = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}

# Acceptance thresholds for a production-quality Wannier model
QUALITY = {"frozen_max_err_meV": 10.0, "frozen_rms_err_meV": 2.0,
           "path_rms_err_meV": 20.0, "max_spread_A2": 15.0}


class Project:
    def __init__(self, workdir: str | Path, cfg: Config, approve=None):
        self.dir = Path(workdir).resolve()
        self.cfg = cfg
        self.state_file = self.dir / "wannagent_state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.approve = approve or (lambda msg: True)
        if cfg.mock:
            from .mock import MockCluster
            self.cluster = MockCluster(self)
        else:
            self.cluster = SlurmCluster()

    # ------------------------------------------------------------------ state
    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    def log(self, action, detail):
        self.state.setdefault("history", []).append(
            {"t": datetime.now().isoformat(timespec="seconds"), "action": action, "detail": detail})
        self.save()

    def step(self, name) -> dict:
        return self.state.setdefault("steps", {}).setdefault(name, {"status": "not_prepared", "attempts": 0})

    def structure(self) -> Structure:
        relax = self.state.get("steps", {}).get("relax", {})
        contcar = self.dir / "relax" / "CONTCAR"
        if relax.get("status") == "done" and contcar.exists() and contcar.stat().st_size > 0:
            return Structure.from_file(str(contcar))
        return Structure.from_file(str(self.dir / "input" / "POSCAR"))

    # ------------------------------------------------------------- actions
    def init(self, structure_path: str, spin: str = "none", relax: bool = False,
             bands: bool = True, projections: dict | None = None) -> dict:
        """Create the project from a POSCAR/CIF. spin: none | collinear | soc."""
        if spin not in ("none", "collinear", "soc"):
            return {"error": "spin must be none, collinear or soc"}
        (self.dir / "input").mkdir(parents=True, exist_ok=True)
        s = st.load_structure(structure_path)
        s.to(filename=str(self.dir / "input" / "POSCAR"), fmt="poscar")
        info = st.describe(s)
        proj = projections or wn.suggest_projections(s)
        plan = (["relax"] if relax else []) + ["scf", "wannier", "wannier90"] + (["bands"] if bands else [])
        self.state.update({"structure_source": str(structure_path), "spin": spin, "plan": plan,
                           "projections": proj, "structure": info, "steps": {}})
        self.log("init", {"spin": spin, "plan": plan})
        warn = []
        if info["possibly_magnetic"] and spin == "none":
            warn.append(f"{info['possibly_magnetic']} present but spin=none: is the material non-magnetic?")
        if any(el in proj for el in ("Gd", "Eu", "Nd", "Sm", "Tb", "Dy")):
            warn.append("4f elements: consider DFT+U and keep f out of the Wannier basis")
        return {"structure": info, "plan": plan, "projections": proj,
                "num_wann": wn.count_wann(s, proj, spin == "soc"), "warnings": warn}

    def set_projections(self, projections: dict) -> dict:
        """Change the trial orbitals, e.g. {"Ni": ["d", "s"], "Se": ["p"]}. Needs a new wannier step."""
        s = self.structure()
        for el in projections:
            if el not in {str(x.specie) for x in s}:
                return {"error": f"{el} not in structure"}
        self.state["projections"] = projections
        for name in ("wannier", "wannier90"):
            if name in self.state.get("steps", {}):
                self.state["steps"][name]["status"] = "not_prepared"
        self.log("set_projections", projections)
        return {"projections": projections, "num_wann": wn.count_wann(s, projections, self.state["spin"] == "soc"),
                "note": "wannier (VASP) step must be prepared and run again: .amn depends on projections"}

    def prepare(self, step: str, incar_overrides: dict | None = None) -> dict:
        if step not in ORDER:
            return {"error": f"step must be one of {ORDER}"}
        need = {"scf": ["relax"], "wannier": ["scf"], "wannier90": ["wannier"], "bands": ["scf"]}[step] \
            if step != "relax" else []
        for dep in need:
            if dep in self.state["plan"] and self.step(dep)["status"] != "done":
                return {"error": f"{dep} must be done before preparing {step} (status: {self.step(dep)['status']})"}
        s, spin = self.structure(), self.state["spin"]
        info = st.describe(s)
        vac = info["vacuum_axes"]
        kg_scf = st.kgrid(s, self.cfg.kppa_scf, vac)
        kg_wan = st.kgrid(s, self.cfg.kppa_wannier, vac)
        rec = self.step(step)

        if step == "wannier90":
            wdir = self.dir / "wannier"
            prefixes = parsers.find_prefixes(wdir)
            missing = [p for p in prefixes if not (wdir / f"{p}.amn").exists()]
            if missing:
                return {"error": f"missing {missing} .amn/.mmn/.eig: wannier VASP step did not write them"}
            res = {"prefixes": prefixes}
            if not self.state.get("windows_set_by_user"):
                res["auto_windows"] = self.auto_windows()
            script = job_script(self.cfg, "wannier90", wdir, spin, prefixes)
            rec.update({"status": "prepared", "dir": str(wdir), "script": str(script), "prefixes": prefixes})
            self.log("prepare", {"step": step})
            return {**res, "script": str(script)}

        win_text, nbands = None, None
        if step in ("wannier", "bands"):
            scf = parsers.parse_vasp_dir(self.dir / "scf")
            ef = scf.get("efermi_eV")
            num_wann = wn.count_wann(s, self.state["projections"], spin == "soc")
            zval = wn.zval_from_potcar(self.dir / "scf" / "POTCAR")
            nbands = wn.suggest_nbands(s, num_wann, spin == "soc", zval)
            if scf.get("nbands"):
                nbands = max(nbands, scf["nbands"])
            self.state.update({"efermi_eV": ef, "num_wann": num_wann, "nbands": nbands})
            kpath = st.kpath_for_wannier(s, vac)
            self.state["kpath"] = kpath
            if step == "wannier":
                win_text = wn.render_win({
                    "num_wann": num_wann, "projections": self.state["projections"],
                    "soc": spin == "soc", "kpath": kpath,
                    # placeholders: VASP only builds .amn/.mmn/.eig; real windows come
                    # from projectability before wannier90.x runs
                    "dis_win_max": (ef or 0) + 8.0, "dis_froz_max": (ef or 0) + 0.5})
        res = prepare_vasp_step(step, s, self.dir, self.cfg, spin, kg_scf, kg_wan,
                                kpath=self.state.get("kpath"), win_text=win_text, nbands=nbands,
                                overrides=incar_overrides, source_dirs={"scf": self.dir / "scf"})
        script = job_script(self.cfg, step, self.dir / step, spin)
        rec.update({"status": "prepared", "dir": res["dir"], "script": str(script),
                    "incar_overrides": incar_overrides or {}})
        self.log("prepare", {"step": step, "overrides": incar_overrides})
        res.update({"kgrid": kg_wan if step == "wannier" else kg_scf, "script": str(script)})
        if step == "wannier":
            res.update({"num_wann": num_wann, "nbands": nbands, "efermi_eV": ef})
        return res

    def submit(self, step: str) -> dict:
        rec = self.step(step)
        if rec["status"] not in ("prepared", "failed"):
            return {"error": f"{step} is {rec['status']}; prepare it first"}
        if rec["attempts"] >= self.cfg.max_resubmits + 1:
            return {"error": f"{step} already tried {rec['attempts']} times; ask the human"}
        if self.cfg.require_approval and not self.approve(f"Submit {step} job ({rec['script']})?"):
            return {"submitted": False, "reason": "human declined"}
        jobid = self.cluster.submit(Path(rec["script"]), step, self.state["spin"])
        rec.update({"status": "submitted", "jobid": jobid, "attempts": rec["attempts"] + 1,
                    "submitted_at": datetime.now().isoformat(timespec="seconds")})
        self.log("submit", {"step": step, "jobid": jobid})
        return {"submitted": True, "jobid": jobid}

    def status(self, step: str | None = None) -> dict:
        names = [step] if step else self.state.get("plan", [])
        out = {}
        for n in names:
            rec = self.step(n)
            if rec["status"] in ("submitted", "running") and rec.get("jobid"):
                sl = self.cluster.status(rec["jobid"])
                rec["slurm_state"] = sl
                if sl == "RUNNING":
                    rec["status"] = "running"
                elif sl in DONE_STATES:
                    rec["status"] = "finished"
                elif sl in FAIL_STATES:
                    rec["status"] = "failed"
            out[n] = {k: rec.get(k) for k in ("status", "jobid", "slurm_state", "attempts")}
        self.save()
        return out

    def wait(self, step: str, max_minutes: float = 60, poll_s: float = 60) -> dict:
        """Block until a job leaves the queue (or max_minutes)."""
        t0 = time.time()
        while True:
            s = self.status(step)[step]
            if s["status"] not in ("submitted", "running"):
                return s
            if time.time() - t0 > max_minutes * 60:
                return {**s, "note": f"still {s['status']} after {max_minutes} min"}
            time.sleep(0 if self.cfg.mock else poll_s)

    def parse(self, step: str) -> dict:
        rec = self.step(step)
        if step == "wannier90":
            wdir = self.dir / "wannier"
            res = {p: parsers.parse_wout(wdir / f"{p}.wout") for p in rec.get("prefixes", ["wannier90"])
                   if (wdir / f"{p}.wout").exists()}
            ok = bool(res) and all(r["finished"] and not r["errors"] for r in res.values())
        else:
            res = parsers.parse_vasp_dir(self.dir / step)
            ok = res["finished"] and not res["errors"] and res.get("scf_converged", True)
            if step == "relax":
                ok = ok and res.get("relax_converged", False)
            if step == "wannier":
                pre = parsers.find_prefixes(self.dir / "wannier")
                have = {p: all((self.dir / "wannier" / f"{p}.{x}").exists() for x in ("amn", "mmn", "eig"))
                        for p in pre}
                res["wannier_files"] = have
                ok = ok and all(have.values())
        rec["status"] = "done" if ok else "failed"
        rec["result"] = res
        self.log("parse", {"step": step, "ok": ok})
        return {"step": step, "ok": ok, "result": res}

    def auto_windows(self) -> dict:
        wdir = self.dir / "wannier"
        out = {}
        for p in parsers.find_prefixes(wdir):
            sug = parsers.suggest_windows(wdir, p, self.state["num_wann"], self.state["efermi_eV"])
            if "error" not in sug:
                wn.update_win_file(wdir / f"{p}.win", {k: sug[k] for k in
                                   ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max")})
            out[p] = sug
        self.state["windows"] = out
        self.log("auto_windows", out)
        return out

    def set_wannier_params(self, changes: dict) -> dict:
        """Edit wannier90.win keys (windows, num_iter, ...) for a wannier90.x rerun. No VASP rerun needed."""
        if "num_wann" in changes or "projections" in changes:
            return {"error": "num_wann/projections change the .amn: use set_projections and rerun the wannier step"}
        wdir = self.dir / "wannier"
        logs = {p: wn.update_win_file(wdir / f"{p}.win", changes) for p in parsers.find_prefixes(wdir)}
        self.state["windows_set_by_user"] = True
        self.step("wannier90")["status"] = "not_prepared"
        self.log("set_wannier_params", changes)
        return {"changed": logs, "next": "prepare and submit wannier90 again"}

    def check_quality(self) -> dict:
        wdir = self.dir / "wannier"
        rep = {"thresholds": QUALITY, "per_spin": {}, "pass": True, "advice": []}
        for p in parsers.find_prefixes(wdir):
            w = parsers.parse_wout(wdir / f"{p}.wout")
            fmin = float(wn.read_win_value(wdir / f"{p}.win", "dis_froz_min") or -1e9)
            fmax = float(wn.read_win_value(wdir / f"{p}.win", "dis_froz_max"))
            fr = parsers.frozen_window_check(wdir, p, fmin, fmax)
            r = {"max_spread_A2": w["max_spread_A2"], "omega_total": w["omega_total"],
                 "disentanglement_converged": w["disentanglement_converged"],
                 "wannierisation_converged": w["wannierisation_converged"], "frozen_window": fr}
            if self.step("bands").get("status") == "done" and (self.dir / "bands" / "vasprun.xml").exists():
                r["path_check"] = parsers.bands_vs_vasp(wdir, self.dir / "bands", p, fmin, fmax)
            rep["per_spin"][p] = r
            adv = rep["advice"]
            if fr["max_err_meV"] is not None and (fr["max_err_meV"] > QUALITY["frozen_max_err_meV"]
                                                  or fr["rms_err_meV"] > QUALITY["frozen_rms_err_meV"]):
                rep["pass"] = False
                adv.append(f"{p}: frozen-window error {fr['max_err_meV']:.1f} meV -> lower dis_froz_max by "
                           "0.2-0.5 eV, or add orbitals (e.g. s on metals) if bands of other character enter")
            if w["max_spread_A2"] and w["max_spread_A2"] > QUALITY["max_spread_A2"]:
                rep["pass"] = False
                adv.append(f"{p}: max spread {w['max_spread_A2']:.1f} A^2 -> projections miss some character "
                           "(check orbital-projected DOS), or the outer window is too wide")
            if not w["disentanglement_converged"]:
                rep["pass"] = False
                adv.append(f"{p}: disentanglement not converged -> dis_num_iter = 5000")
            if not w["wannierisation_converged"]:
                adv.append(f"{p}: wannierisation hit num_iter (fine if spreads are small; else num_iter = 5000)")
            pc = r.get("path_check")
            if pc and pc["rms_err_meV"] and pc["rms_err_meV"] > QUALITY["path_rms_err_meV"]:
                rep["pass"] = False
                adv.append(f"{p}: off-grid band error {pc['rms_err_meV']:.0f} meV -> denser Wannier k-grid "
                           "(raise kppa_wannier) or check for spurious states")
        self.state["quality"] = rep
        self.log("check_quality", {"pass": rep["pass"]})
        return rep

    def collect(self) -> dict:
        """Copy the final model files to <workdir>/final for downstream codes (WannierBerri, your OHC/SHC codes)."""
        wdir, fin = self.dir / "wannier", self.dir / "final"
        fin.mkdir(exist_ok=True)
        copied = []
        for p in parsers.find_prefixes(wdir):
            for suf in ("_hr.dat", "_r.dat", "_tb.dat", "_centres.xyz", "_wsvec.dat", ".win", ".wout",
                        "_band.dat", "_band.kpt"):
                f = wdir / f"{p}{suf}"
                if f.exists():
                    shutil.copy(f, fin / f.name)
                    copied.append(f.name)
        return {"final_dir": str(fin), "files": copied}

    def summary(self) -> dict:
        return {"dir": str(self.dir), "spin": self.state.get("spin"), "plan": self.state.get("plan"),
                "projections": self.state.get("projections"), "num_wann": self.state.get("num_wann"),
                "efermi_eV": self.state.get("efermi_eV"),
                "steps": {k: {"status": v.get("status"), "attempts": v.get("attempts")}
                          for k, v in self.state.get("steps", {}).items()},
                "quality_pass": self.state.get("quality", {}).get("pass")}
