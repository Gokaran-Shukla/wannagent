"""Rule-based pipeline: the same tools as the agent, with fixed recovery rules.

Use it when you want reproducible high-throughput runs, or as the baseline that
the LLM agent must beat (a good interview talking point: agent vs rules).
"""
from __future__ import annotations

from .workflow import Project

FIXES = [  # (error substring, INCAR overrides to add on retry)
    ("EDDDAV", {"ALGO": "All", "NCORE": 1}),
    ("Sub-Space-Matrix", {"ALGO": "All"}),
    ("BRMIX", {"AMIX": 0.1, "BMIX": 0.0001, "AMIX_MAG": 0.4, "BMIX_MAG": 0.0001}),
    ("VERY BAD NEWS", {"NCORE": 1}),
    ("ZBRENT", {"IBRION": 1, "POTIM": 0.2}),
]


def fix_for(result: dict, previous: dict) -> dict | None:
    for err in result.get("errors", []):
        for key, ov in FIXES:
            if key in err["match"]:
                return {**previous, **ov}
    if result.get("finished") and result.get("scf_converged") is False:
        return {**previous, "NELM": 400, "ALGO": "All"}
    return None


def run_vasp_step(p: Project, step: str, say=print) -> bool:
    overrides = {}
    for attempt in range(p.cfg.max_resubmits + 1):
        r = p.prepare(step, overrides or None)
        if "error" in r:
            say(f"[{step}] cannot prepare: {r['error']}")
            return False
        s = p.submit(step)
        if not s.get("submitted"):
            say(f"[{step}] not submitted: {s}")
            return False
        say(f"[{step}] submitted job {s['jobid']} (attempt {attempt + 1})")
        p.wait(step, max_minutes=48 * 60)
        res = p.parse(step)
        if res["ok"]:
            say(f"[{step}] done")
            return True
        new = fix_for(res["result"], overrides)
        if new is None or new == overrides:
            say(f"[{step}] failed with no known fix: {res['result'].get('errors') or res['result'].get('log_tail')}")
            return False
        say(f"[{step}] failed; retrying with {new}")
        overrides = new
    return False


def run_wannier90(p: Project, say=print, max_window_tries=3) -> bool:
    for attempt in range(max_window_tries):
        r = p.prepare("wannier90")
        if "error" in r:
            say(f"[wannier90] {r['error']}")
            return False
        if "auto_windows" in r:
            say(f"[wannier90] projectability windows: { {k: v.get('dis_froz_max') for k, v in r['auto_windows'].items()} }")
        p.submit("wannier90")
        p.wait("wannier90", max_minutes=600)
        if not p.parse("wannier90")["ok"]:
            say("[wannier90] run failed; see .wout/.werr")
            return False
        q = p.check_quality()
        for pre, v in q["per_spin"].items():
            fw = v["frozen_window"]
            say(f"[quality] {pre}: max spread {v['max_spread_A2']:.2f} A^2, frozen-window error "
                f"max {fw['max_err_meV']:.2f} / rms {fw['rms_err_meV']:.2f} meV")
        if q["pass"]:
            return True
        say(f"[quality] FAIL: {q['advice']}")
        if p.state.get("windows_set_by_user"):
            p.state["windows_set_by_user"] = False      # first fallback: projectability windows
            p.step("wannier90")["status"] = "not_prepared"
            say("[wannier90] retry with projectability-based windows")
            continue
        lowered = {}
        for pre in q["per_spin"]:
            from .wannier import read_win_value
            fmax = float(read_win_value(p.dir / "wannier" / f"{pre}.win", "dis_froz_max"))
            lowered = {"dis_froz_max": round(fmax - 0.3, 4)}
        p.set_wannier_params(lowered)
        say(f"[wannier90] retry with {lowered}")
    return False


def run_pipeline(p: Project, say=print) -> dict:
    plan = p.state["plan"]
    order = [s for s in ("relax", "scf", "bands", "wannier") if s in plan]
    for step in order:
        if p.step(step)["status"] == "done":
            continue
        if not run_vasp_step(p, step, say):
            return {"ok": False, "stopped_at": step, "summary": p.summary()}
    ok = run_wannier90(p, say)
    if ok:
        say(f"[collect] {p.collect()['files']}")
    return {"ok": ok, "summary": p.summary()}
