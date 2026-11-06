"""Advanced track A1-A7. Each check returns a score plus whether it counts.
Lower or higher is better per challenge (see META in registry.py)."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from . import assets
from .beginner import _strip_trailing_measure_on
from .core import (Spec, SubmissionError, count_named, cx_count, cx_edges,
                   equiv_up_to_output_perm, parse, placeholder_qubits,
                   reduced_fidelity_dm, split_final_measurements, state_of,
                   substitute, unitary_of)


def scored(score: float, msg: str, valid: bool = True, **extra) -> dict:
    return {"score": float(score), "valid": bool(valid), "message": msg, **extra}


def one(data, spec):
    if isinstance(data, list):
        if len(data) != 1:
            raise SubmissionError("This challenge takes one circuit.")
        data = data[0]
    return parse(data, spec)


def _check_edges(qc, edges, chip):
    bad = sorted(e for e in cx_edges(qc) if e not in edges)
    if bad:
        raise SubmissionError(f"Two-qubit gate between {bad[0][0]} and {bad[0][1]}, which aren't connected on the {chip}. "
                              f"Allowed pairs: {sorted(edges)}.")


@lru_cache(maxsize=None)
def _baseline(which: str) -> int:
    from qiskit.transpiler import CouplingMap
    if which == "a1":
        tgt, edges = assets.a1_target(), assets.LINE4
    else:
        tgt, edges = assets.a7_target(), assets.HEAVYHEX6
    cmap = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    best = None
    for seed in range(8):
        t = transpile(tgt, coupling_map=cmap, basis_gates=["cx", "u"], optimization_level=3,
                      initial_layout=list(range(tgt.num_qubits)), seed_transpiler=seed)
        n = t.count_ops().get("cx", 0)
        best = n if best is None else min(best, n)
    return best


# ---------- A1 circuit golf ----------

def a1(data, ctx):
    qc = one(data, Spec(max_qubits=4, min_qubits=4))
    _check_edges(qc, assets.LINE4, "linear chip 0-1-2-3")
    if equiv_up_to_output_perm(unitary_of(qc), unitary_of(assets.a1_target()), 4) is None:
        return scored(math.inf, "Not equivalent to the target circuit (up to a final relabelling of qubits).", valid=False)
    n = cx_count(qc)
    base = _baseline("a1")
    beat = " You beat Qiskit's optimizer." if n < base else ""
    return scored(n, f"{n} CNOTs. Qiskit optimization_level=3 gets {base}.{beat}", cx=n)


# ---------- A2 QAOA MaxCut ----------

@lru_cache(maxsize=None)
def _cuts():
    v = assets.a2_cut_values()
    return v, float(v.max())


def a2(data, ctx):
    if not isinstance(data, dict) or "gammas" not in data or "betas" not in data:
        raise SubmissionError('Submit {"gammas": [...], "betas": [...]} (one of each per layer).')
    try:
        g = [float(x) for x in data["gammas"]]
        b = [float(x) for x in data["betas"]]
    except Exception:
        raise SubmissionError("gammas and betas must be lists of numbers.") from None
    if len(g) != len(b) or not 1 <= len(g) <= assets.A2_MAX_P:
        raise SubmissionError(f"Need the same number of gammas and betas, between 1 and {assets.A2_MAX_P} layers.")
    if not all(math.isfinite(x) and abs(x) < 100 for x in g + b):
        raise SubmissionError("Angles must be finite numbers below 100 in size.")
    sv = Statevector(assets.a2_circuit(g, b))
    cuts, cmax = _cuts()
    expected = float(np.dot(sv.probabilities(), cuts))
    ratio = expected / cmax
    return scored(ratio, f"Expected cut {expected:.3f} of max {cmax:.0f}: ratio {ratio:.4f} with p={len(g)}.",
                  tiebreak=len(g))


# ---------- A3 VQE ----------

@lru_cache(maxsize=None)
def _a3_ground():
    H = assets.tfim(assets.A3_N)
    return H, float(np.linalg.eigvalsh(H.to_matrix()).min())


def a3(data, ctx):
    qc = one(data, Spec(max_qubits=assets.A3_N, min_qubits=assets.A3_N))
    n = cx_count(qc)
    if n > assets.A3_CX_CAP:
        raise SubmissionError(f"{n} CNOTs is over the cap of {assets.A3_CX_CAP}.")
    H, e0 = _a3_ground()
    e = float(np.real(state_of(qc).expectation_value(H)))
    err = e - e0
    score = err + assets.A3_CX_WEIGHT * n
    return scored(score, f"Energy {e:.5f} (exact {e0:.5f}, error {err:.5f}) with {n} CNOTs. "
                  f"Score = error + {assets.A3_CX_WEIGHT} x CNOTs.", energy=e, cx=n)


# ---------- A4 noisy estimation (hidden score) ----------

def a4_ideal(team_seed: int) -> float:
    from qiskit.quantum_info import SparsePauliOp
    sv = Statevector(assets.a4_target(team_seed))
    return float(np.real(sv.expectation_value(SparsePauliOp(assets.A4_OBSERVABLE))))


def a4(data, ctx):
    if isinstance(data, dict):
        data = data.get("value")
    try:
        v = float(data)
    except Exception:
        raise SubmissionError("Submit your estimate as a single number.") from None
    if not math.isfinite(v):
        raise SubmissionError("Estimate must be a finite number.")
    err = abs(v - a4_ideal(ctx["team_seed"]))
    return scored(err, f"Estimate {v:.5f} recorded. Scores stay hidden until the reveal; your last submission counts.",
                  hidden=True, estimate=v)


def a4_run(circuits, shots, ctx):
    from qiskit_aer import AerSimulator
    parsed = []
    for text in circuits:
        qc = parse(text, Spec(max_qubits=assets.A4_MAX_QUBITS))
        if cx_count(qc) > assets.A4_CX_CAP:
            raise SubmissionError(f"Each circuit may have at most {assets.A4_CX_CAP} CNOTs.")
        if qc.num_clbits == 0 or count_named(qc, "measure") == 0:
            qc.measure_all()
        parsed.append(qc)
    sim = AerSimulator(noise_model=assets.a4_noise_model())
    tq = transpile(parsed, sim, optimization_level=0)
    res = sim.run(tq, shots=shots, seed_simulator=ctx.get("run_seed")).result()
    return [res.get_counts(i) for i in range(len(tq))]


# ---------- A5 Trotter ----------

@lru_cache(maxsize=None)
def _a5_exact():
    return assets.a5_exact_state()


def a5(data, ctx):
    qc = one(data, Spec(max_qubits=assets.A5_N, min_qubits=assets.A5_N))
    n = cx_count(qc)
    if n > assets.A5_CX_CAP:
        raise SubmissionError(f"{n} CNOTs is over the cap of {assets.A5_CX_CAP}.")
    f = float(abs(np.vdot(_a5_exact().data, state_of(qc).data)) ** 2)
    score = f - assets.A5_CX_WEIGHT * n
    return scored(score, f"Fidelity {f:.5f} with {n} CNOTs. Score = fidelity - {assets.A5_CX_WEIGHT} x CNOTs.",
                  fidelity=f, cx=n)


# ---------- A6 repetition code under noise ----------

def _cardinal_states():
    out = []
    for build in ("0", "1", "+", "-", "r", "l"):
        c = QuantumCircuit(1)
        if build == "1": c.x(0)
        if build in "+-": c.h(0)
        if build == "-": c.z(0)
        if build in "rl": c.h(0); c.s(0)
        if build == "l": c.z(0)
        out.append(c)
    return out


def _idle_channel():
    from qiskit_aer.noise import pauli_error
    q = QuantumCircuit(3, name="idle")
    inst = pauli_error([("X", assets.A6_P_FLIP), ("I", 1 - assets.A6_P_FLIP)]).to_instruction()
    for i in range(3):
        q.append(inst, [i])
    return q


def _a6_error(body: QuantumCircuit, shots: int) -> float:
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="density_matrix", noise_model=assets.a6_noise_model())
    variant = substitute(body, "idle", _idle_channel())
    errs = []
    for prep in _cardinal_states():
        c = variant.copy_empty_like()
        c.compose(prep, qubits=[0], inplace=True)
        c.compose(variant, inplace=True)
        c.save_density_matrix(qubits=[0], label="rho")
        tc = transpile(c, sim, optimization_level=0)
        rho = sim.run(tc, shots=shots, seed_simulator=7).result().data()["rho"]
        errs.append(1 - reduced_fidelity_dm(rho, Statevector(prep)))
    return float(np.mean(errs))


@lru_cache(maxsize=None)
def a6_baseline() -> float:
    from qiskit.circuit import Gate
    q = QuantumCircuit(3)
    q.append(Gate("idle", 3, []), [0, 1, 2])
    return _a6_error(q, shots=1)


def a6(data, ctx):
    qc = one(data, Spec(max_qubits=5, min_qubits=3, allow_dynamic=True, placeholders={"idle": 3}))
    locs = placeholder_qubits(qc, "idle")
    if count_named(qc, "idle") != 1 or len(locs) != 1 or locs[0] != [0, 1, 2]:
        raise SubmissionError("Use the idle gate exactly once, on qubits [0, 1, 2], outside any if_test.")
    body = _strip_trailing_measure_on(qc, 0)
    dynamic = count_named(body, "if_else") > 0 or count_named(body, "measure") > 0
    err = _a6_error(body, shots=2000 if dynamic else 1)
    base = a6_baseline()
    valid = err < base
    verdict = "Beats" if valid else "Does not beat"
    return scored(err, f"Logical error {err:.5f} averaged over 6 input states. {verdict} the unprotected qubit ({base:.5f}).",
                  valid=valid)


# ---------- A7 routing ----------

def a7(data, ctx):
    qc = one(data, Spec(max_qubits=6, min_qubits=6))
    _check_edges(qc, assets.HEAVYHEX6, "heavy-hex fragment")
    if equiv_up_to_output_perm(unitary_of(qc), unitary_of(assets.a7_target()), 6) is None:
        return scored(math.inf, "Not equivalent to the target (allowing a final relabelling of qubits). "
                      "Keep the initial layout trivial: logical qubit i starts on physical qubit i.", valid=False)
    n = cx_count(qc)
    base = _baseline("a7")
    bonus = n < base
    return scored(n, f"{n} CNOTs. Qiskit optimization_level=3 gets {base}." + (" Bonus: you beat it." if bonus else ""),
                  bonus=bonus, cx=n)


CHECKS = {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6, "a7": a7}
