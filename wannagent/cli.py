"""Command line:

  wannagent init     --structure POSCAR --dir proj --spin soc [--relax] [--no-bands]
  wannagent pipeline --dir proj                     # rule-based, no LLM
  wannagent agent    --dir proj --goal "..."        # LLM agent (needs ANTHROPIC_API_KEY)
  wannagent agent    --dir proj --resume            # continue after long jobs
  wannagent status   --dir proj
  wannagent quality  --dir proj

All commands take --config cluster.yaml (see examples/config_ibex.yaml) and --mock.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .workflow import Project


def _approve(msg: str) -> bool:
    if not sys.stdin.isatty():
        print(f"[approval needed, no terminal] {msg} -> declined")
        return False
    return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wannagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "pipeline", "agent", "status", "quality"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--config")
    ap.add_argument("--mock", action="store_true", help="fake cluster for testing")
    ap.add_argument("--yes", action="store_true", help="approve all submissions (no prompts)")
    ap.add_argument("--structure")
    ap.add_argument("--spin", default="none", choices=["none", "collinear", "soc"])
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--no-bands", action="store_true")
    ap.add_argument("--projections", help='JSON, e.g. \'{"Ni":["d"],"Se":["p"]}\'')
    ap.add_argument("--goal")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    cfg.mock = cfg.mock or a.mock
    if a.yes:
        cfg.require_approval = False
    p = Project(a.dir, cfg, approve=_approve)
    show = lambda x: print(json.dumps(x, indent=2, default=str))

    if a.command == "init":
        proj = json.loads(a.projections) if a.projections else None
        show(p.init(a.structure, a.spin, a.relax, not a.no_bands, proj))
    elif a.command == "pipeline":
        from .pipeline import run_pipeline
        res = run_pipeline(p)
        show(res["summary"])
        sys.exit(0 if res["ok"] else 1)
    elif a.command == "agent":
        from .agent import Agent
        goal = a.goal
        if not goal and not a.resume:
            goal = (f"Build a Wannier tight-binding model for the project in {p.dir}. "
                    + ("The project is initialised; continue from its current state."
                       if p.state else "Initialise it first."))
        res = Agent(p, model=a.model).run(goal, resume=a.resume)
        print("\n" + "=" * 70 + f"\nsuccess: {res.get('success')}\n{res.get('report')}")
    elif a.command == "status":
        p.status()
        show(p.summary())
    elif a.command == "quality":
        show(p.check_quality())


if __name__ == "__main__":
    main()
