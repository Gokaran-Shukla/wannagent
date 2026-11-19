# wannagent

An AI agent that runs the full **VASP + Wannier90** workflow on a SLURM cluster, plus a rule-based pipeline that uses the same tools without an LLM.

```
structure ──► [relax] ──► scf ──► [bands] ──► wannier (VASP: .amn .mmn .eig)
                                                  │
                     projectability ◄─────────────┘
                     energy windows
                          │
                          ▼
                    wannier90.x ──► check_quality ──pass──► final/ (hr.dat, r.dat, tb.dat)
                          ▲               │
                          └──── fix ◄─fail┘   (lower dis_froz_max / add orbitals / denser k)
```

## Why this design

* **The LLM never writes INCARs or shell commands.** It can only call 13 typed tools
  (`prepare_step`, `submit_step`, `parse_step`, `set_wannier_params`, `check_quality`, ...).
  Every tool validates its inputs, and errors go back to the LLM as results it can read, so a bad call never crashes the run.
* **The quality check is quantitative.** Wannier H(R) is evaluated at the DFT k-points and
  compared with DFT eigenvalues inside the frozen window (target <10 meV max, <2 meV rms).
  An optional VASP line-mode run also checks off-grid accuracy along the high-symmetry path.
* **Energy windows come from band projectability** (sum over trial orbitals of |A_mn(k)|^2 from `.amn`),
  not hand-tuned: outer window = bands with p > 0.05, frozen window = up to the energy where
  every band still has p > 0.85 and no k has more than num_wann bands.
* **Re-running Wannier90 is cheap.** Changing the windows edits `wannier90.win` in place and reruns `wannier90.x`, with no new VASP run.
  Changing projections triggers a new wannier VASP step automatically.
* **A human approves submissions.** Each `sbatch` asks y/N unless you pass `--yes`.
* **Runs can be resumed.** State lives in `wannagent_state.json` and the agent conversation in
  `agent_conversation.json`, so a 20-hour SCF doesn't need a 20-hour agent process.
* Handles `spin: none | collinear | soc`: collinear runs give separate `wannier90.up` and `.dn` models, both checked;
  SOC uses `vasp_ncl`, sets `spinors = true` and doubles num_wann.
* Builds the k-path for the bands plot from Setyawan–Curtarolo, mapped onto **your** cell by an integer lattice
  mapping. It works for rotated, conventional or supercell inputs, and keeps only in-plane segments for slabs.

## Install

```bash
conda create -n wannagent python=3.12 -y && conda activate wannagent
pip install -e .           # numpy, pymatgen, pyyaml, anthropic
pytest -q tests/           # 10 tests on the mock cluster, ~2 s
```

## Try it with no cluster (mock mode)

```bash
wannagent init     --dir si_test --mock --structure examples/Si.vasp
wannagent pipeline --dir si_test --config examples/config_mock.yaml
```
The mock produces fake VASP and Wannier90 outputs from a random tight-binding model plus an
entangled band at E_F + 1.5 eV. `config_mock.yaml` also makes SCF fail once (EDDDAV), so you
can watch the recovery.

## Real runs on Ibex

1. Edit `examples/config_ibex.yaml` so its `modules`, `partition` and `potcar_dir` match your `submit.sh`.
   VASP must be built with the Wannier90 interface (`-DVASP2WANNIER90`, Wannier90 v3 library).
2. Rule-based, no LLM:
   ```bash
   wannagent init --dir CrSb --config examples/config_ibex.yaml --structure CrSb.vasp \
                  --spin collinear --projections '{"Cr":["d"],"Sb":["p"]}'
   wannagent pipeline --dir CrSb --config examples/config_ibex.yaml
   ```
3. Agent:
   ```bash
   export ANTHROPIC_API_KEY=...
   wannagent agent --dir CrSb --config examples/config_ibex.yaml \
     --goal "Collinear AFM CrSb from CrSb.vasp. Wannier model with Cr d and Sb p for SOT and orbital Hall work."
   # after long jobs, or a disconnect:
   wannagent agent --dir CrSb --config examples/config_ibex.yaml --resume
   ```
   Run it inside `tmux` on the login node; each tool call only writes small files or runs sbatch, so the login node does no heavy computing.
4. `wannagent status --dir CrSb` and `wannagent quality --dir CrSb` work at any time.

The final model lands in `CrSb/final/`, ready for WannierBerri or your CuPy OHC/SHC codes.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Cluster/code settings (YAML) |
| `structure.py` | Structure info, slab detection, k-grids, k-path mapped to input cell |
| `wannier.py` | Default projections, num_wann, NBANDS, `.win` rendering and in-place editing |
| `vasp_inputs.py` | INCAR/KPOINTS/POTCAR for relax, scf, wannier, bands |
| `slurm.py` | Job scripts, sbatch, squeue/sacct |
| `parsers.py` | OUTCAR/OSZICAR/vasp.out errors, `.wout`, `.eig`, `.amn`, `hr.dat`; quality checks; projectability windows |
| `workflow.py` | `Project`: state machine; every method doubles as an agent tool |
| `pipeline.py` | Rule-based runner with fixed recovery rules (the baseline the agent must beat) |
| `agent.py` | Tool schemas, system prompt, Anthropic tool-use loop, resume, logs |
| `mock.py` | Fake cluster for tests |

## Before trusting it on production

Tested so far: the logic end to end on the mock cluster only. Check these on a first real
run (Si, then a magnetic case):

* VASP 6 writes `wannier90.win` from the `WANNIER90_WIN` tag, including cell and k-points. For
  VASP 5, set `vasp_version: 5` and check that the cell and k-point blocks were added.
* `bands_vs_vasp` reads `bands/vasprun.xml`; it has not yet been run on real output.
* The default projections are a heuristic. For metals, `s` on the transition metal often improves
  disentanglement. Check the orbital-projected DOS.
* The VASP error-to-fix table in `parsers.py` and `pipeline.py` covers common failures only. Extend it as you meet new ones.

## Ideas to extend (good portfolio items)

* A `run_wannierberri` tool: OHC/SHC straight after `collect`, so the agent can screen a list of materials.
* Batch mode: one agent manages N materials, prioritising the queue and resubmitting failures.
* Score the agent against `pipeline.py` on 20 materials (jobs used, wall time, pass rate).
