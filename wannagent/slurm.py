"""SLURM job scripts, submission and status. MockCluster in mock.py has the same interface."""
from __future__ import annotations

import subprocess
from pathlib import Path


def job_script(cfg, kind: str, step_dir: Path, spin: str, prefixes=("wannier90",)) -> Path:
    s = cfg.slurm
    is_w90 = kind == "wannier90"
    hdr = [
        "#!/bin/bash",
        f"#SBATCH --job-name=wa_{step_dir.parent.name[:10]}_{kind}",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --nodes={s.nodes}",
        f"#SBATCH --ntasks={s.wannier_ntasks if is_w90 else s.ntasks}",
        f"#SBATCH --cpus-per-task={s.cpus_per_task}",
        f"#SBATCH --time={s.wannier_time if is_w90 else s.time}",
        "#SBATCH --output=slurm-%j.out",
    ]
    if s.mem:
        hdr.append(f"#SBATCH --mem={s.mem}")
    if s.account:
        hdr.append(f"#SBATCH --account={s.account}")
    if s.constraint:
        hdr.append(f"#SBATCH --constraint={s.constraint}")
    body = ["", "set -e", f"cd {step_dir}"]
    body += [f"module load {m}" for m in s.modules]
    body += list(s.pre_commands)
    body.append("export OMP_NUM_THREADS=1")
    if is_w90:
        for p in prefixes:
            body.append(f"{cfg.wannier90} {p}")
    else:
        if kind in ("wannier", "bands"):
            body.append("[ -f CHGCAR ] || cp ../scf/CHGCAR .")
        exe = cfg.vasp_ncl if spin == "soc" else cfg.vasp_std
        body.append(f"{exe} > vasp.out 2>&1")
    body.append("echo WANNAGENT_DONE")
    path = step_dir / f"job_{kind}.sh"
    path.write_text("\n".join(hdr + body) + "\n")
    path.chmod(0o755)
    return path


class SlurmCluster:
    def submit(self, script: Path, kind: str, spin: str) -> str:
        r = subprocess.run(["sbatch", "--parsable", script.name], cwd=script.parent,
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")
        return r.stdout.strip().split(";")[0]

    def status(self, jobid: str) -> str:
        r = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%T"], capture_output=True, text=True)
        st = r.stdout.strip()
        if st:
            return st                          # PENDING / RUNNING / CONFIGURING ...
        r = subprocess.run(["sacct", "-n", "-X", "-j", jobid, "-o", "State"], capture_output=True, text=True)
        st = r.stdout.strip().split()
        return st[0] if st else "UNKNOWN"      # COMPLETED / FAILED / TIMEOUT / CANCELLED

    def cancel(self, jobid: str) -> str:
        subprocess.run(["scancel", jobid])
        return "CANCELLED"
