"""Fixed problem instances and the oracle/placeholder gates.

The same constructions are mirrored in client/fallfest.py so participants can
test locally. The grader always swaps in its own version of a placeholder
gate, so editing the local copy gains nothing.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


# ---------- beginner oracles ----------

def bv_oracle(secret: str) -> QuantumCircuit:
    """4 input qubits (0-3) + ancilla (4). secret is written like a Qiskit
    count string: secret[-1] is qubit 0."""
    qc = QuantumCircuit(5, name="oracle")
    for i, b in enumerate(reversed(secret)):
        if b == "1":
            qc.cx(i, 4)
    return qc


def grover_oracle(marked: str) -> QuantumCircuit:
    """Phase oracle on 2 qubits: flips the sign of |marked> (Qiskit order)."""
    qc = QuantumCircuit(2, name="oracle")
    zeros = [i for i, b in enumerate(reversed(marked)) if b == "0"]
    if zeros:
        qc.x(zeros)
    qc.cz(0, 1)
    if zeros:
        qc.x(zeros)
    return qc


def bitflip(k: int | None) -> QuantumCircuit:
    """The 'noise' placeholder for B13: X on code qubit k, or nothing."""
    qc = QuantumCircuit(3, name="noise")
    if k is not None:
        qc.x(k)
    return qc


# ---------- B14 / A7 routing targets ----------

def b14_target() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 1)
    qc.cx(0, 2)
    qc.rz(1.3, 2)
    qc.cx(0, 3)
    qc.h(3)
    return qc


LINE4 = {(0, 1), (1, 2), (2, 3)}
LINE5 = {(0, 1), (1, 2), (2, 3), (3, 4)}
# Heavy-hex fragment: a degree-3 node (2) with three arms, as on IBM chips.
HEAVYHEX6 = {(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)}


def a7_target() -> QuantumCircuit:
    qc = QuantumCircuit(6)
    qc.h(range(6))
    pairs = [(0, 4), (1, 5), (3, 0), (4, 5), (2, 4), (0, 1), (5, 3), (1, 4)]
    for k, (a, b) in enumerate(pairs):
        qc.cx(a, b)
        qc.rz(0.3 + 0.17 * k, b)
    qc.ry(0.4, range(6))
    return qc


# ---------- A1 circuit golf ----------

def a1_target() -> QuantumCircuit:
    """A deliberately bloated 4-qubit circuit (all CX already on the line)."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1); qc.cx(0, 1); qc.cx(0, 1)          # = one CX
    qc.swap(1, 2); qc.cx(2, 3); qc.swap(1, 2)      # = CX(1,3) the long way
    qc.rz(0.5, 3)
    qc.cx(2, 3); qc.cx(2, 3)                        # = identity
    qc.ccx(0, 1, 2)
    qc.cx(1, 2); qc.rz(0.25, 2); qc.cx(1, 2)
    qc.h(3); qc.cz(2, 3); qc.h(3)                   # = CX(2,3)
    qc.ry(0.9, 0)
    qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)          # = SWAP(0,1)
    return qc


# ---------- A2 QAOA MaxCut ----------

A2_EDGES = [(0, 1), (0, 4), (0, 7), (1, 2), (1, 8), (2, 3), (2, 6), (3, 4), (3, 9),
            (4, 5), (5, 6), (5, 9), (6, 7), (7, 8), (8, 9)]
A2_N = 10
A2_MAX_P = 5


def a2_cut_values() -> np.ndarray:
    idx = np.arange(2 ** A2_N)
    bits = (idx[:, None] >> np.arange(A2_N)) & 1
    return sum((bits[:, a] != bits[:, b]).astype(float) for a, b in A2_EDGES)


def a2_circuit(gammas, betas) -> QuantumCircuit:
    qc = QuantumCircuit(A2_N)
    qc.h(range(A2_N))
    for g, b in zip(gammas, betas):
        for u, v in A2_EDGES:
            qc.rzz(-g, u, v)  # exp(+i g Z Z / 2); sign convention documented in the notebook
        qc.rx(2 * b, range(A2_N))
    return qc


# ---------- A3 / A5 Ising models ----------

def tfim(n: int, J: float = 1.0, h: float = 1.0) -> SparsePauliOp:
    """H = -J sum Z_i Z_{i+1} - h sum X_i (open chain)."""
    terms = []
    for i in range(n - 1):
        terms.append(("ZZ", [i, i + 1], -J))
    for i in range(n):
        terms.append(("X", [i], -h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


A3_N = 6
A3_CX_CAP = 60
A3_CX_WEIGHT = 0.005

A5_N = 5
A5_T = 1.5
A5_CX_WEIGHT = 0.002
A5_CX_CAP = 150


def a5_exact_state():
    from scipy.linalg import expm
    from qiskit.quantum_info import Statevector
    H = tfim(A5_N).to_matrix()
    psi0 = np.zeros(2 ** A5_N, dtype=complex); psi0[0] = 1
    return Statevector(expm(-1j * A5_T * H) @ psi0)


# ---------- A4 noisy estimation ----------

A4_SHOT_BUDGET = 400_000
A4_MAX_QUBITS = 5
A4_CX_CAP = 80


def a4_target(team_seed: int) -> QuantumCircuit:
    """Per-team circuit (different angles per team, so answers can't be shared).
    Ideal <ZZZZ> is about 0.7-0.8; raw noisy runs land near 0.3."""
    rng = np.random.default_rng(team_seed)
    qc = QuantumCircuit(4)
    for _ in range(3):
        for i in range(4):
            qc.ry(rng.uniform(0.05, 0.35), i)
        qc.cx(0, 1); qc.cx(2, 3); qc.cx(1, 2)
        for i in range(4):
            qc.rz(rng.uniform(0, 1), i)
        qc.cx(1, 2); qc.cx(2, 3); qc.cx(0, 1)
    return qc


A4_OBSERVABLE = "ZZZZ"


def a4_noise_model():
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["u", "u1", "u2", "u3", "rz", "ry", "rx", "sx", "x", "h", "s", "sdg", "t", "tdg", "y", "z", "p"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ["cx", "cz", "ecr", "swap", "rzz", "cp", "crz"])
    nm.add_all_qubit_readout_error(ReadoutError([[0.96, 0.04], [0.06, 0.94]]))
    return nm


# ---------- A6 repetition code under noise ----------

A6_P_FLIP = 0.10
A6_P2 = 0.005
A6_P1 = 0.0005


def a6_noise_model():
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(A6_P1, 1), ["u", "u1", "u2", "u3", "rz", "ry", "rx", "sx", "x", "h", "s", "sdg", "t", "tdg", "y", "z", "p"])
    nm.add_all_qubit_quantum_error(depolarizing_error(A6_P2, 2), ["cx", "cz", "ecr", "swap", "rzz", "cp", "crz", "cy", "ch"])
    nm.add_all_qubit_quantum_error(depolarizing_error(2 * A6_P2, 3), ["ccx", "cswap", "ccz"])
    return nm
