"""Reference solutions (should pass / score well) and common wrong answers
(should fail with a useful hint). Used by test_grader.py and handy for mentors."""
import numpy as np
from qiskit import QuantumCircuit

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "client"))
from fallfest import (bv_oracle, grover_oracle, idle_gate, noise_gate,  # noqa: E402
                      to_qasm)


def c(n, cl=0):
    return QuantumCircuit(n, cl)


# ---------- correct ----------

def b1():
    q = c(1, 1); q.x(0); q.measure(0, 0); return q

def b2():
    q = c(1, 1); q.h(0); q.measure(0, 0); return q

def b3():
    q = c(1, 1); q.h(0); q.h(0); q.measure(0, 0); return q

def b4():
    q = c(1, 1); q.ry(2 * np.arcsin(np.sqrt(0.25)), 0); q.measure(0, 0); return q

def b5():
    q = c(2, 2); q.h(0); q.cx(0, 1); q.measure([0, 1], [0, 1]); return q

def b6():
    q = c(2, 2); q.h(0); q.cx(0, 1); q.x(1); q.measure([0, 1], [0, 1]); return q

def b7():
    out = []
    for z, x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        q = c(2)
        if x: q.x(1)
        q.h(0)
        if z: q.z(0)
        q.cx(0, 1)
        out.append(q)
    return out

def b8():
    q = c(5); q.h(0); q.cx(0, 1); q.cx(0, 2); q.cx(1, 3); q.cx(2, 4); return q

def b8_chain():
    q = c(5); q.h(0)
    for i in range(4): q.cx(i, i + 1)
    return q

def b9():
    q = c(1, 1); q.h(0); q.measure(0, 0); return q

def b10():
    q = c(5, 4); q.x(4); q.h(range(5))
    q.append(bv_oracle("1011"), range(5))
    q.h(range(4)); q.measure(range(4), range(4)); return q

def b11():
    q = c(2, 2); q.h([0, 1])
    q.append(grover_oracle("10"), [0, 1])
    q.h([0, 1]); q.x([0, 1]); q.cz(0, 1); q.x([0, 1]); q.h([0, 1])
    q.measure([0, 1], [0, 1]); return q

def b12():
    q = c(3, 2)
    q.h(1); q.cx(1, 2); q.cx(0, 1); q.h(0)
    q.measure(0, 0); q.measure(1, 1)
    with q.if_test((q.clbits[1], 1)): q.x(2)
    with q.if_test((q.clbits[0], 1)): q.z(2)
    return q

def b13():
    q = c(5, 2)
    q.cx(0, 1); q.cx(0, 2)
    q.append(noise_gate(), [0, 1, 2])
    q.cx(0, 3); q.cx(1, 3); q.cx(1, 4); q.cx(2, 4)
    q.measure(3, 0); q.measure(4, 1)
    with q.if_test((q.cregs[0], 1)): q.x(0)   # s=01: qubit 0
    with q.if_test((q.cregs[0], 3)): q.x(1)   # s=11: qubit 1
    with q.if_test((q.cregs[0], 2)): q.x(2)   # s=10: qubit 2
    q.cx(0, 2); q.cx(0, 1)
    return q

def b13_coherent():
    """Measurement-free version using Toffolis."""
    q = c(3)
    q.cx(0, 1); q.cx(0, 2)
    q.append(noise_gate(), [0, 1, 2])
    q.cx(0, 1); q.cx(0, 2); q.ccx(2, 1, 0)
    return q

def b14():
    q = c(4); q.h(0); q.cx(0, 1); q.ry(0.7, 1)
    q.swap(0, 1)                     # logical 0 now on physical 1
    q.cx(1, 2); q.rz(1.3, 2)
    q.swap(1, 2)                     # logical 0 now on physical 2
    q.cx(2, 3); q.h(3)
    return q


# ---------- wrong ----------

WRONG = {
    "b1": [lambda: (lambda q: (q.measure(0, 0), q)[1])(c(1, 1)), b2],
    "b2": [b1],
    "b3": [b2, lambda: c(1)],
    "b4": [b2],
    "b5": [lambda: (lambda q: (q.h([0, 1]), q)[1])(c(2)), b6],
    "b6": [b5],
    "b7": [lambda: [b7()[0], b7()[0], b7()[2], b7()[3]]],
    "b8": [lambda: (lambda q: (q.h(0), q.cx(0, 1), q)[2])(c(5))],
    "b9": [lambda: (lambda q: (q.measure(0, 0), q)[1])(c(1, 1))],
    "b10": [lambda: (lambda q: (q.h(range(4)), q.append(bv_oracle("1"), range(5)), q.h(range(4)), q.measure(range(4), range(4)), q)[4])(c(5, 4))],
    "b11": [lambda: (lambda q: (q.h([0, 1]), q.append(grover_oracle("10"), [0, 1]), q.measure([0, 1], [0, 1]), q)[3])(c(2, 2))],
    "b12": [lambda: (lambda q: (q.swap(0, 2), q)[1])(c(3))],
    "b13": [lambda: (lambda q: (q.append(noise_gate(), [0, 1, 2]), q)[1])(c(3))],
    "b14": [lambda: (lambda q: (q.h(0), q.cx(0, 1), q.ry(0.7, 1), q.cx(0, 2), q.rz(1.3, 2), q.cx(0, 3), q.h(3), q)[7])(c(4))],
}

PASS = {f"b{i}": globals()[f"b{i}"] for i in range(1, 15)}
PASS_EXTRA = {"b8": [b8_chain], "b13": [b13_coherent]}


def enc(x):
    return [to_qasm(q) for q in x] if isinstance(x, list) else to_qasm(x)
