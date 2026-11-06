"""Challenge catalogue and the single entry point the server calls in a worker process."""
from __future__ import annotations

import traceback

from qiskit import qasm2

from . import advanced, assets, beginner
from .core import SubmissionError

META = {
    "b1": ("Flip it", "beginner"),
    "b2": ("Quantum coin", "beginner"),
    "b3": ("Undo the magic", "beginner"),
    "b4": ("Loaded coin", "beginner"),
    "b5": ("States that agree", "beginner"),
    "b6": ("States that disagree", "beginner"),
    "b7": ("Bell zoo", "beginner"),
    "b8": ("GHZ fan-out", "beginner"),
    "b9": ("Plus or minus?", "beginner"),
    "b10": ("Find the secret number", "beginner"),
    "b11": ("Find the marked card", "beginner"),
    "b12": ("Teleport it", "beginner"),
    "b13": ("Fix the flip", "beginner"),
    "b14": ("Fit on the chip", "beginner"),
    "a1": ("Circuit golf", "low"),
    "a2": ("QAOA MaxCut", "high"),
    "a3": ("Ground state hunt", "low"),
    "a4": ("Beat the noise", "low"),
    "a5": ("Trotter tradeoff", "high"),
    "a6": ("Beat the physical error rate", "low"),
    "a7": ("Route it", "low"),
}

DESCRIPTIONS = {
    "b14": "Make this circuit run on a linear chip 0-1-2-3 (CNOTs only between neighbors), with at most 2 SWAPs.",
    "a1": "Rewrite this circuit for a linear chip 0-1-2-3 with as few CNOTs as possible. A final relabelling of qubits is allowed.",
    "a2": f"QAOA on the fixed 10-node graph below, up to p={assets.A2_MAX_P}. Submit gammas and betas.",
    "a3": f"Prepare the ground state of the {assets.A3_N}-qubit transverse-field Ising chain H = -sum Z_i Z_(i+1) - sum X_i. "
          f"Score = energy error + {assets.A3_CX_WEIGHT} x CNOTs (cap {assets.A3_CX_CAP}).",
    "a4": f"Estimate <{assets.A4_OBSERVABLE}> of your team's circuit using run_noisy(). Budget {assets.A4_SHOT_BUDGET:,} shots.",
    "a5": f"Starting from |00000>, prepare exp(-iHt)|00000> for the {assets.A5_N}-qubit Ising chain (same H as A3), t={assets.A5_T}. "
          f"Score = fidelity - {assets.A5_CX_WEIGHT} x CNOTs.",
    "a6": f"Protect qubit 0 through the idle gate (10% bit-flip per qubit). Gates are noisy too. Lower logical error wins; must beat the bare qubit.",
    "a7": "Make this 6-qubit circuit legal on the heavy-hex fragment with as few CNOTs as possible. Initial layout must be trivial.",
}

CHECKS = {**beginner.CHECKS, **advanced.CHECKS}


def kind(cid: str) -> str:
    return "pass" if META[cid][1] == "beginner" else "score"


def info(cid: str, team_seed: int) -> dict:
    title, mode = META[cid]
    out = {"id": cid, "title": title, "kind": kind(cid),
           "better": None if mode == "beginner" else mode,
           "description": DESCRIPTIONS.get(cid, "")}
    targets = {"b14": assets.b14_target, "a1": assets.a1_target, "a7": assets.a7_target}
    if cid in targets:
        out["target_qasm"] = qasm2.dumps(targets[cid]())
    if cid == "a4":
        out["target_qasm"] = qasm2.dumps(assets.a4_target(team_seed))
        out["observable"] = assets.A4_OBSERVABLE
        out["shot_budget"] = assets.A4_SHOT_BUDGET
    if cid == "a2":
        out["edges"] = assets.A2_EDGES
    if cid == "a7":
        out["coupling_map"] = sorted(assets.HEAVYHEX6)
    if cid in ("a1", "b14"):
        out["coupling_map"] = sorted(assets.LINE4)
    return out


def grade(cid: str, data, ctx: dict) -> dict:
    """Runs inside a worker process. Never raises: returns {'error': ...} instead."""
    try:
        return CHECKS[cid](data, ctx)
    except SubmissionError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "The grader hit an internal problem with this submission. Tell a mentor.",
                "internal": traceback.format_exc()[-2000:]}


def run_noisy(circuits, shots: int, ctx: dict) -> dict:
    try:
        return {"counts": advanced.a4_run(circuits, shots, ctx)}
    except SubmissionError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "Simulator problem. Tell a mentor.", "internal": traceback.format_exc()[-2000:]}


def warm_up():
    """Precompute cached baselines so the first participant doesn't wait."""
    advanced._baseline("a1"); advanced._baseline("a7")
    advanced._cuts(); advanced._a3_ground(); advanced._a5_exact(); advanced.a6_baseline()
    return True