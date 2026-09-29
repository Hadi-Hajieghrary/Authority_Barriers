"""Trial logs: numpy arrays + JSON sidecar with every constant a figure must report."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .filter_system import DIAG_FIELDS


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@dataclass
class TrialLog:
    t_plant: np.ndarray        # (K,)
    x_plant: np.ndarray        # (nx, K)
    t_diag: np.ndarray         # (M,)
    diag: np.ndarray           # (len(DIAG_FIELDS), M)
    cable: np.ndarray          # (2N+3, M) tensions, stretch, a_meas at the filter ticks
    cmd: np.ndarray            # (3N, M)
    att_err: np.ndarray        # (N, M) attitude tracking error at the ticks (NaN if absent)
    taut: np.ndarray           # (6+6N, M) projected taut-cable state at the ticks
    meta: dict = field(default_factory=dict)
    termination: str = "horizon"

    def d(self, name: str) -> np.ndarray:
        return self.diag[DIAG_FIELDS.index(name)]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path) + ".npz", t_plant=self.t_plant, x_plant=self.x_plant, t_diag=self.t_diag,
                            diag=self.diag, cable=self.cable, cmd=self.cmd, att_err=self.att_err, taut=self.taut)
        meta = dict(self.meta)
        meta["termination"] = self.termination
        meta["diag_fields"] = DIAG_FIELDS
        meta["git_sha"] = git_sha()
        Path(str(path) + ".json").write_text(json.dumps(meta, indent=1, default=_json_default))

    @staticmethod
    def load(path: Path) -> "TrialLog":
        path = str(path)
        for suf in (".npz", ".json"):
            if path.endswith(suf):
                path = path[: -len(suf)]
        z = np.load(path + ".npz")
        meta = json.loads(Path(path + ".json").read_text())
        return TrialLog(z["t_plant"], z["x_plant"], z["t_diag"], z["diag"], z["cable"], z["cmd"], z["att_err"], z["taut"],
                        meta, meta.get("termination", "horizon"))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
