"""LLM agent that drives the VASP + Wannier90 workflow through tool calls.

The LLM never writes INCARs or shell commands itself: it can only call the
typed tools below, each a Project method with validation. Every sbatch goes
through a human-approval gate unless require_approval is false. The whole
conversation is saved, so `wannagent agent --resume` continues after a long
VASP job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .workflow import Project

SYSTEM = """You are a computational materials physicist operating a VASP + Wannier90 workflow
on a SLURM cluster through tools. Goal: a production-quality maximally localised Wannier
tight-binding model (hr.dat) for the user's material, ready for transport / Berry-curvature /
orbital-Hall calculations.

Workflow: init -> [relax] -> scf -> [bands] -> wannier (VASP writes .amn/.mmn/.eig) ->
wannier90 (wannier90.x) -> check_quality -> collect. Steps follow prepare -> submit ->
wait -> parse.

Physics rules you must follow:
- Projections: d on transition metals, p on p-block anions, sp3 on C/Si/Ge. 4f elements:
  keep f out of the basis unless the user asks. Explain every projection change.
- spin: none | collinear (VASP writes wannier90.up/.dn; both must pass) | soc (vasp_ncl,
  spinors, num_wann doubled). Ask the user if magnetism or SOC is unclear; do not guess.
- Energy windows are absolute energies (eV), not relative to E_F. prepare(wannier90)
  sets them from band projectability; trust that unless quality fails.
- A model passes only if check_quality passes: DFT eigenvalues inside the frozen window
  reproduced to <10 meV max and <2 meV rms, spreads < 15 A^2, disentanglement converged.
- Fix order when quality fails: (1) lower dis_froz_max in 0.2-0.5 eV steps; (2) add missing
  orbital character (e.g. s on metals) via set_projections, which needs a new wannier VASP
  step; (3) raise kppa_wannier only if the off-grid path check fails.
- VASP failures: read parse output errors and their suggested_fix, then prepare again with
  incar_overrides. Never change ENCUT, functional, or POTCARs without telling the user.
- Use the fewest jobs. Never resubmit a step that is done. Never invent numbers: quote
  tool outputs. Finish with finish(), giving E_F, num_wann, windows, errors and spreads.
"""


def _schema(props: dict, req: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": req}


STEP = {"type": "string", "enum": ["relax", "scf", "wannier", "wannier90", "bands"]}
TOOLS = [
    {"name": "init_project", "description": "Create the project from a structure file (POSCAR or CIF).",
     "input_schema": _schema({"structure_path": {"type": "string"},
                              "spin": {"type": "string", "enum": ["none", "collinear", "soc"]},
                              "relax": {"type": "boolean"}, "bands": {"type": "boolean"},
                              "projections": {"type": "object", "description": "e.g. {\"Ni\": [\"d\"], \"Se\": [\"p\"]}"}},
                             ["structure_path", "spin"])},
    {"name": "set_projections", "description": "Replace trial orbitals. Requires rerunning the wannier VASP step.",
     "input_schema": _schema({"projections": {"type": "object"}}, ["projections"])},
    {"name": "prepare_step", "description": "Write inputs and job script for a step. incar_overrides: INCAR tags to set (null value deletes a tag).",
     "input_schema": _schema({"step": STEP, "incar_overrides": {"type": "object"}}, ["step"])},
    {"name": "submit_step", "description": "sbatch a prepared step (a human may need to approve).",
     "input_schema": _schema({"step": STEP}, ["step"])},
    {"name": "job_status", "description": "SLURM status of one step, or all if step omitted.",
     "input_schema": _schema({"step": STEP}, [])},
    {"name": "wait_for_job", "description": "Block until the job leaves the queue or max_minutes passes.",
     "input_schema": _schema({"step": STEP, "max_minutes": {"type": "number"}}, ["step"])},
    {"name": "parse_step", "description": "Parse outputs of a finished step: convergence, E_F, errors with suggested fixes.",
     "input_schema": _schema({"step": STEP}, ["step"])},
    {"name": "set_wannier_params", "description": "Edit wannier90.win keys (dis_froz_max, dis_win_max, dis_froz_min, dis_win_min, dis_num_iter, num_iter) for a wannier90.x rerun. Values are absolute eV.",
     "input_schema": _schema({"changes": {"type": "object"}}, ["changes"])},
    {"name": "check_quality", "description": "Quantitative Wannier model checks with pass/fail and advice.",
     "input_schema": _schema({}, [])},
    {"name": "read_file", "description": "Read the last lines of a file inside the project (e.g. wannier/wannier90.wout, scf/vasp.out).",
     "input_schema": _schema({"path": {"type": "string"}, "tail": {"type": "integer"}}, ["path"])},
    {"name": "project_summary", "description": "Current state of every step.", "input_schema": _schema({}, [])},
    {"name": "collect_results", "description": "Copy final hr.dat, r.dat, tb.dat, centres to final/.", "input_schema": _schema({}, [])},
    {"name": "finish", "description": "End the session with a report for the user.",
     "input_schema": _schema({"report": {"type": "string"}, "success": {"type": "boolean"}}, ["report", "success"])},
]


class Agent:
    def __init__(self, project: Project, client=None, model: str | None = None,
                 max_turns: int = 60, max_wait_minutes: float = 120, say=print):
        self.p = project
        self.model = model or os.environ.get("WANNAGENT_MODEL", "claude-sonnet-5-5")
        self.client = client
        self.max_turns = max_turns
        self.max_wait = max_wait_minutes
        self.say = say
        self.conv_file = project.dir / "agent_conversation.json"
        self.log_file = project.dir / "agent_log.jsonl"
        self.done = None

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY
        return self.client

    # ------------------------------------------------------------ tools
    def call_tool(self, name: str, a: dict):
        p = self.p
        try:
            if name == "init_project":
                return p.init(a["structure_path"], a["spin"], a.get("relax", False), a.get("bands", True),
                              a.get("projections"))
            if name == "set_projections":
                return p.set_projections(a["projections"])
            if name == "prepare_step":
                return p.prepare(a["step"], a.get("incar_overrides"))
            if name == "submit_step":
                return p.submit(a["step"])
            if name == "job_status":
                return p.status(a.get("step"))
            if name == "wait_for_job":
                return p.wait(a["step"], min(a.get("max_minutes", self.max_wait), self.max_wait))
            if name == "parse_step":
                return p.parse(a["step"])
            if name == "set_wannier_params":
                return p.set_wannier_params(a["changes"])
            if name == "check_quality":
                return p.check_quality()
            if name == "read_file":
                f = (p.dir / a["path"]).resolve()
                if p.dir not in f.parents:
                    return {"error": "path outside project"}
                if not f.exists():
                    return {"error": "no such file"}
                return {"text": "\n".join(f.read_text(errors="ignore").splitlines()[-int(a.get("tail", 60)):])}
            if name == "project_summary":
                return p.summary()
            if name == "collect_results":
                return p.collect()
            if name == "finish":
                self.done = a
                return {"ok": True}
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # tools never crash the agent; the LLM sees the error
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------- loop
    def _log(self, kind, data):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({"kind": kind, "data": data}, default=str) + "\n")

    def run(self, goal: str | None = None, resume: bool = False) -> dict:
        if resume and self.conv_file.exists():
            messages = json.loads(self.conv_file.read_text())
            messages.append({"role": "user", "content": goal or
                             "Resuming: jobs may have finished. Check status and continue."})
        else:
            messages = [{"role": "user", "content": goal}]
        for turn in range(self.max_turns):
            resp = self._client().messages.create(model=self.model, max_tokens=4096, system=SYSTEM,
                                                  tools=TOOLS, messages=messages)
            content = [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in resp.content]
            messages.append({"role": "assistant", "content": content})
            for b in content:
                if b["type"] == "text" and b["text"].strip():
                    self.say(f"\n[agent] {b['text'].strip()}")
            if resp.stop_reason != "tool_use":
                break
            results = []
            for b in content:
                if b["type"] != "tool_use":
                    continue
                self.say(f"  -> {b['name']}({json.dumps(b['input'])[:160]})")
                out = self.call_tool(b["name"], b["input"])
                txt = json.dumps(out, default=str)
                if len(txt) > 8000:
                    txt = txt[:8000] + ' ... [truncated]'
                self._log("tool", {"name": b["name"], "input": b["input"], "output": out})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": txt,
                                "is_error": isinstance(out, dict) and "error" in out})
            messages.append({"role": "user", "content": results})
            self.conv_file.write_text(json.dumps(messages, default=str))
            if self.done:
                break
        self.conv_file.write_text(json.dumps(messages, default=str))
        return self.done or {"success": False, "report": "stopped: turn limit or no finish() call"}
