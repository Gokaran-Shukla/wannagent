"""Cluster and code settings, read from a YAML file (see examples/config_ibex.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class SlurmConfig:
    partition: str = "batch"
    nodes: int = 1
    ntasks: int = 40
    cpus_per_task: int = 1
    time: str = "24:00:00"
    mem: str = ""                       # e.g. "200G"; empty = node default
    account: str = ""
    constraint: str = ""
    modules: list[str] = field(default_factory=list)
    pre_commands: list[str] = field(default_factory=list)   # e.g. conda activate, export OMP_NUM_THREADS=1
    wannier_ntasks: int = 1             # wannier90.x is cheap; serial is usually fine
    wannier_time: str = "02:00:00"


@dataclass
class Config:
    potcar_dir: str = "/home/shuklagn/potpaw_PBE"
    vasp_std: str = "srun vasp_std"
    vasp_ncl: str = "srun vasp_ncl"
    wannier90: str = "wannier90.x"
    vasp_version: int = 6               # 6: inputs passed via WANNIER90_WIN tag
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    mock: bool = False                  # True: fake cluster, outputs generated instantly
    mock_failures: list[str] = field(default_factory=list)  # mock: steps that fail on 1st attempt
    require_approval: bool = True       # ask a human before every sbatch
    max_resubmits: int = 3
    kppa_scf: int = 3000                # k-points per reciprocal atom for SCF
    kppa_wannier: int = 1500            # usually coarser grid is fine for Wannier projections

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        slurm = SlurmConfig(**data.pop("slurm", {}))
        return cls(slurm=slurm, **data)

    def to_dict(self) -> dict:
        return asdict(self)
