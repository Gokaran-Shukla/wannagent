"""End-to-end tests on the mock cluster:  pytest -q tests/"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from scipy.spatial.transform import Rotation

from wannagent import Config, Project
from wannagent.agent import Agent
from wannagent.parsers import read_hr, wannier_eigs
from wannagent.pipeline import run_pipeline, run_wannier90
from wannagent.structure import kpath_for_wannier

EX = Path(__file__).resolve().parents[1] / "examples"
warnings.filterwarnings("ignore")


def mock_cfg(**kw):
    return Config(mock=True, require_approval=False, kppa_wannier=300, **kw)


def test_pipeline_si_with_scf_failure(tmp_path):
    p = Project(tmp_path / "si", mock_cfg(mock_failures=["scf"]))
    p.init(str(EX / "Si.vasp"))
    res = run_pipeline(p, say=lambda *_: None)
    assert res["ok"]
    assert p.step("scf")["attempts"] == 2                       # EDDDAV -> ALGO=All retry
    assert "ALGO = All" in (p.dir / "scf" / "INCAR").read_text()
    assert p.state["quality"]["pass"]
    assert (p.dir / "final" / "wannier90_hr.dat").exists()


def test_collinear_writes_up_and_dn(tmp_path):
    p = Project(tmp_path / "ni", mock_cfg())
    p.init(str(EX / "Ni_fcc.vasp"), spin="collinear", projections={"Ni": ["s", "d"]})
    assert run_pipeline(p, say=lambda *_: None)["ok"]
    assert set(p.state["quality"]["per_spin"]) == {"wannier90.up", "wannier90.dn"}
    assert "ISPIN = 2" in (p.dir / "scf" / "INCAR").read_text()


def test_soc_doubles_num_wann_and_uses_ncl(tmp_path):
    p = Project(tmp_path / "mos2", mock_cfg())
    r = p.init(str(EX / "MoS2_bulk.vasp"), spin="soc")
    assert r["num_wann"] == 2 * (2 * 5 + 4 * 3)
    assert run_pipeline(p, say=lambda *_: None)["ok"]
    assert "vasp_ncl" in (p.dir / "wannier" / "job_wannier.sh").read_text()
    assert "spinors = true" in (p.dir / "wannier" / "wannier90.win").read_text()


def test_bad_window_is_detected_and_repaired(tmp_path):
    p = Project(tmp_path / "si", mock_cfg())
    p.init(str(EX / "Si.vasp"))
    run_pipeline(p, say=lambda *_: None)
    p.set_wannier_params({"dis_froz_max": 8.0})                 # into the entangled band
    p.prepare("wannier90"); p.submit("wannier90"); p.wait("wannier90"); p.parse("wannier90")
    assert not p.check_quality()["pass"]
    assert run_wannier90(p, say=lambda *_: None)
    assert p.check_quality()["pass"]


@pytest.mark.parametrize("fname", ["Si.vasp", "MoS2_bulk.vasp"])
def test_kpath_independent_of_cell_orientation(fname):
    s = Structure.from_file(str(EX / fname))
    R = Rotation.from_euler("zyx", [23, 41, -17], degrees=True).as_matrix()
    s2 = Structure(Lattice(s.lattice.matrix @ R.T), s.species, s.frac_coords)
    a = kpath_for_wannier(s)
    b = kpath_for_wannier(s2)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-5)


def test_hr_roundtrip_reproduces_mock_dft(tmp_path):
    p = Project(tmp_path / "si", mock_cfg())
    p.init(str(EX / "Si.vasp"))
    run_pipeline(p, say=lambda *_: None)
    R, deg, H = read_hr(p.dir / "wannier" / "wannier90_hr.dat")
    e = wannier_eigs(R, deg, H, np.zeros((1, 3)))
    assert e.shape == (1, 8) and np.all(np.diff(e[0]) >= -1e-12)


# --------------------------------------------------------------- agent loop
class ScriptedLLM:
    """Stands in for the Anthropic client: replays a plausible tool-call trajectory
    and reacts to tool results (it reads the scf parse and applies the suggested fix)."""

    def __init__(self, structure):
        self.structure = structure
        self.i = 0
        self.messages = self

    def _tool(self, name, inp):
        self.i += 1
        return {"type": "tool_use", "id": f"t{self.i}", "name": name, "input": inp}

    def create(self, model, max_tokens, system, tools, messages):
        last = messages[-1]["content"]
        results = [json.loads(r["content"]) for r in last] if isinstance(last, list) else []
        n_asst = sum(m["role"] == "assistant" for m in messages)
        seq = [
            [self._tool("init_project", {"structure_path": self.structure, "spin": "none", "bands": False})],
            [self._tool("prepare_step", {"step": "scf"})],
            [self._tool("submit_step", {"step": "scf"}), self._tool("wait_for_job", {"step": "scf"})],
            [self._tool("parse_step", {"step": "scf"})],
        ]
        if n_asst < len(seq):
            blocks = seq[n_asst]
        elif n_asst == len(seq):
            fix = {}
            if not results[0]["ok"]:                            # react to the failure
                fix = {"ALGO": "All", "NCORE": 1}
            blocks = [self._tool("prepare_step", {"step": "scf", "incar_overrides": fix}),
                      self._tool("submit_step", {"step": "scf"}),
                      self._tool("wait_for_job", {"step": "scf"}),
                      self._tool("parse_step", {"step": "scf"})]
        else:
            k = n_asst - len(seq) - 1
            rest = [
                [self._tool("prepare_step", {"step": "wannier"})],
                [self._tool("submit_step", {"step": "wannier"}), self._tool("wait_for_job", {"step": "wannier"}),
                 self._tool("parse_step", {"step": "wannier"})],
                [self._tool("prepare_step", {"step": "wannier90"})],
                [self._tool("submit_step", {"step": "wannier90"}), self._tool("wait_for_job", {"step": "wannier90"}),
                 self._tool("parse_step", {"step": "wannier90"})],
                [self._tool("check_quality", {})],
                [self._tool("collect_results", {})],
            ]
            if k < len(rest):
                blocks = rest[k]
            else:
                q = next(r for r in results)
                blocks = [{"type": "text", "text": "Model complete."},
                          self._tool("finish", {"report": "done", "success": True})]
        return type("R", (), {"content": blocks, "stop_reason": "tool_use"})()


def test_agent_loop_recovers_and_finishes(tmp_path):
    p = Project(tmp_path / "agent_si", mock_cfg(mock_failures=["scf"]))
    a = Agent(p, client=ScriptedLLM(str(EX / "Si.vasp")), say=lambda *_: None)
    res = a.run("Build a Wannier model of Si")
    assert res["success"]
    assert p.state["quality"]["pass"]
    log = [json.loads(l) for l in (p.dir / "agent_log.jsonl").read_text().splitlines()]
    assert any(e["data"]["name"] == "parse_step" and not e["data"]["output"]["ok"] for e in log)


def test_approval_gate_blocks_submission(tmp_path):
    cfg = mock_cfg()
    cfg.require_approval = True
    p = Project(tmp_path / "gate", cfg, approve=lambda msg: False)
    p.init(str(EX / "Si.vasp"))
    p.prepare("scf")
    assert p.submit("scf") == {"submitted": False, "reason": "human declined"}


def test_agent_cannot_read_outside_project(tmp_path):
    p = Project(tmp_path / "x", mock_cfg())
    assert "error" in Agent(p, client=object()).call_tool("read_file", {"path": "../../etc/passwd"})
