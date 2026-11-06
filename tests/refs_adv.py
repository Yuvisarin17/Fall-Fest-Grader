"""Reference submissions for the advanced track."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from scipy.optimize import minimize

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "client"))
from grader import assets  # noqa: E402
from fallfest import idle_gate, to_qasm  # noqa: E402


def _route(tgt, edges):
    cm = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    return transpile(tgt, coupling_map=cm, basis_gates=["cx", "u"], optimization_level=3,
                     initial_layout=list(range(tgt.num_qubits)), seed_transpiler=0)


def a1():
    return _route(assets.a1_target(), assets.LINE4)


def a7():
    return _route(assets.a7_target(), assets.HEAVYHEX6)


def a2():
    from qiskit.quantum_info import Statevector
    cuts = assets.a2_cut_values()
    f = lambda x: -float(np.dot(Statevector(assets.a2_circuit(x[:2], x[2:])).probabilities(), cuts))
    best = min((minimize(f, x0, method="COBYLA", options={"maxiter": 300})
                for x0 in np.random.default_rng(0).uniform(0, 1, (4, 4))), key=lambda r: r.fun)
    return {"gammas": list(best.x[:2]), "betas": list(best.x[2:])}


def a3():
    n = assets.A3_N
    def ansatz(t):
        q = QuantumCircuit(n)
        q.ry(t[:n], range(n)) if False else [q.ry(t[i], i) for i in range(n)]
        for i in range(n - 1): q.cx(i, i + 1)
        for i in range(n): q.ry(t[n + i], i)
        return q
    from qiskit.quantum_info import Statevector
    H = assets.tfim(n)
    f = lambda t: float(np.real(Statevector(ansatz(t)).expectation_value(H)))
    r = minimize(f, np.full(2 * n, 0.3), method="L-BFGS-B")
    return ansatz(r.x)


def a5(steps=4):
    n, T = assets.A5_N, assets.A5_T
    dt = T / steps
    q = QuantumCircuit(n)
    for _ in range(steps):
        for i in range(n - 1): q.rzz(-2 * dt, i, i + 1)
        q.rx(-2 * dt, range(n))
    return q


def a6_dynamic():
    q = QuantumCircuit(5, 2)
    q.cx(0, 1); q.cx(0, 2)
    q.append(idle_gate(), [0, 1, 2])
    q.cx(0, 3); q.cx(1, 3); q.cx(1, 4); q.cx(2, 4)
    q.measure(3, 0); q.measure(4, 1)
    with q.if_test((q.cregs[0], 1)): q.x(0)
    with q.if_test((q.cregs[0], 3)): q.x(1)
    with q.if_test((q.cregs[0], 2)): q.x(2)
    q.cx(0, 2); q.cx(0, 1)
    return q


def a6_coherent():
    q = QuantumCircuit(3)
    q.cx(0, 1); q.cx(0, 2)
    q.append(idle_gate(), [0, 1, 2])
    q.cx(0, 1); q.cx(0, 2); q.ccx(2, 1, 0)
    return q


def a6_cheat_classical_copy():
    """Measure q0 before idling and rewrite it after: fine for |0>,|1>, wrecks |+>."""
    q = QuantumCircuit(3, 1)
    q.measure(0, 0)
    q.append(idle_gate(), [0, 1, 2])
    q.reset(0)
    with q.if_test((q.clbits[0], 1)): q.x(0)
    return q
