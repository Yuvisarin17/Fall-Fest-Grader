"""Qiskit Fall Fest 2026 grader client.

In Colab:
    !pip install -q qiskit qiskit-aer requests
    !wget -q https://<grader-host>/fallfest.py
    from fallfest import *
    login("YOUR-TEAM-TOKEN", server="http://<grader-host>")

Then build a circuit and call:
    submit("b1", qc)
"""
from __future__ import annotations

import os

import time

import numpy as np
import requests
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2, qasm3
from qiskit.circuit import ControlFlowOp

SERVER = os.environ.get("FALLFEST_SERVER", "http://localhost:8000")
RAISE_ERRORS = False  # True makes submit() raise instead of printing (used by tests)


class GraderError(RuntimeError):
    pass
_TOKEN = os.environ.get("FALLFEST_TOKEN")

__all__ = ["QuantumCircuit", "QuantumRegister", "ClassicalRegister", "np",
           "login", "submit", "run_noisy", "challenge_info", "my_progress",
           "bv_oracle", "grover_oracle", "noise_gate", "idle_gate", "a1_target", "a7_target",
           "b14_target", "a4_target", "SERVER", "counts", "tfim",
           "A2_EDGES", "A2_MAX_P", "LINE4", "HEAVYHEX6"]

# Fixed problem data (same as the grader's)
A2_EDGES = [(0, 1), (0, 4), (0, 7), (1, 2), (1, 8), (2, 3), (2, 6), (3, 4), (3, 9),
            (4, 5), (5, 6), (5, 9), (6, 7), (7, 8), (8, 9)]
A2_MAX_P = 5
LINE4 = [(0, 1), (1, 2), (2, 3)]
HEAVYHEX6 = [(0, 1), (1, 2), (2, 3), (2, 5), (3, 4)]


def tfim(n: int, J: float = 1.0, h: float = 1.0):
    """H = -J sum Z_i Z_(i+1) - h sum X_i on an open chain of n qubits (SparsePauliOp)."""
    from qiskit.quantum_info import SparsePauliOp
    terms = [("ZZ", [i, i + 1], -J) for i in range(n - 1)] + [("X", [i], -h) for i in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def counts(qc: QuantumCircuit, shots: int = 1000) -> dict:
    """Run a circuit on a perfect simulator on your own machine and return the counts."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    c = qc.copy()
    if c.num_clbits == 0:
        c.measure_all()
    sim = AerSimulator()
    return dict(sorted(sim.run(transpile(c, sim), shots=shots).result().get_counts().items()))


def login(token: str, server: str | None = None):
    global _TOKEN, SERVER
    _TOKEN = token.strip()
    if server:
        SERVER = server.rstrip("/")
    try:
        r = _get("/api/me")
    except GraderError as e:
        print(f"❌ Login failed: {e}")
        return
    print(f"Logged in as team '{r['team']}'.")


def _headers():
    if not _TOKEN:
        raise GraderError("Run the setup cell with login(...) first.")
    return {"Authorization": f"Bearer {_TOKEN}"}


def _get(path):
    try:
        r = requests.get(SERVER + path, headers=_headers(), timeout=60)
    except requests.RequestException:
        raise GraderError("Can't reach the grader. Check your internet, or tell a mentor.") from None
    return _handle(r)


def _post(path, payload):
    for attempt in range(4):
        try:
            r = requests.post(SERVER + path, json=payload, headers=_headers(), timeout=120)
        except requests.RequestException:
            raise GraderError("Can't reach the grader. Check your internet, or tell a mentor.") from None
        if r.status_code == 429 and "Slow down" in r.text and attempt < 3:
            time.sleep(float(r.headers.get("Retry-After", 3)))
            continue
        return _handle(r)


def _handle(r):
    try:
        body = r.json()
    except Exception:
        raise GraderError(f"Server error ({r.status_code}). Tell a mentor.") from None
    if r.status_code >= 400:
        raise GraderError(body.get("detail", body))
    return body


def _has_control_flow(qc: QuantumCircuit) -> bool:
    return any(isinstance(i.operation, ControlFlowOp) for i in qc.data)


def to_qasm(qc: QuantumCircuit) -> str:
    if qc.num_parameters:
        raise ValueError("Your circuit still has unbound parameters. Use qc.assign_parameters(...) first.")
    if _has_control_flow(qc):
        return qasm3.dumps(qc)
    try:
        return qasm2.dumps(qc)
    except Exception:
        return qasm3.dumps(qc)


def _encode(x):
    if isinstance(x, QuantumCircuit):
        return to_qasm(x)
    if isinstance(x, (list, tuple)) and x and all(isinstance(c, QuantumCircuit) for c in x):
        return [to_qasm(c) for c in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _encode(v) for k, v in x.items()}
    return x


class Result(dict):
    """The grader's reply. Prints nothing extra in a notebook; use it like a dict."""
    def _ipython_display_(self):
        pass


def submit(challenge: str, answer):
    """answer: a QuantumCircuit, a list of circuits, a dict of numbers, or a number."""
    try:
        res = Result(_post("/api/submit", {"challenge": challenge.lower(), "data": _encode(answer)}))
    except (GraderError, ValueError, TypeError) as e:
        if RAISE_ERRORS:
            raise
        print(f"❌ {challenge.upper()}: {e}")
        return Result(passed=False, error=str(e))
    if "passed" in res:
        mark = "✅" if res["passed"] else "❌"
        print(f"{mark} {challenge.upper()}: {res['message']}")
        if res.get("badge"):
            print(f"🏅 Badge earned: {res['badge']}")
    else:
        score = res.get("score")
        if not res.get("valid", True):
            extra = "" if score is None else f" (score {score:.6g}, not counted)"
            print(f"❌ {challenge.upper()}: {res['message']}{extra}")
        elif score is None:
            print(f"📨 {challenge.upper()}: {res['message']}")
        else:
            print(f"📈 {challenge.upper()} score: {score:.6g}. {res['message']}")
    return res


def run_noisy(circuits, shots: int = 4000):
    """A4 only: run circuits on the shared noisy simulator. Returns a list of counts dicts."""
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    try:
        res = _post("/api/run", {"challenge": "a4", "circuits": [to_qasm(c) for c in circuits], "shots": shots})
    except (GraderError, ValueError) as e:
        if RAISE_ERRORS:
            raise
        print(f"❌ {e}")
        return []
    print(f"Shots left in your budget: {res['shots_left']:,}")
    return res["counts"]


def challenge_info(challenge: str):
    return _get(f"/api/challenge/{challenge.lower()}")


def my_progress():
    return _get("/api/me")


# ---------- oracles and placeholders (local versions for testing) ----------

def bv_oracle(secret: str) -> "Gate":
    """B10: a 5-qubit oracle hiding a 4-bit secret. Qubits 0-3 are inputs, qubit 4 is the ancilla.
    The grader swaps in its own random secret, so any secret works for local testing."""
    qc = QuantumCircuit(5, name="oracle")
    for i, b in enumerate(reversed(secret)):
        if b == "1":
            qc.cx(i, 4)
    return qc.to_gate()


def grover_oracle(marked: str) -> "Gate":
    """B11: flips the sign of the 2-qubit state |marked>. The grader tests all four."""
    qc = QuantumCircuit(2, name="oracle")
    zeros = [i for i, b in enumerate(reversed(marked)) if b == "0"]
    if zeros:
        qc.x(zeros)
    qc.cz(0, 1)
    if zeros:
        qc.x(zeros)
    return qc.to_gate()


def noise_gate(flip: int | None = None) -> "Gate":
    """B13: put this on qubits [0, 1, 2]. Locally it flips qubit `flip` (or nothing).
    The grader tries no flip and a flip on each of the three qubits."""
    qc = QuantumCircuit(3, name="noise")
    if flip is not None:
        qc.x(flip)
    return qc.to_gate()


def idle_gate() -> "Gate":
    """A6: marks where your encoded qubits sit idle. The grader applies an independent
    bit-flip with probability 10% to each of qubits 0, 1, 2 here."""
    qc = QuantumCircuit(3, name="idle")
    return qc.to_gate()


def _fetch_circuit(cid: str) -> QuantumCircuit:
    text = challenge_info(cid)["target_qasm"]
    return qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def b14_target() -> QuantumCircuit:
    return _fetch_circuit("b14")


def a1_target() -> QuantumCircuit:
    return _fetch_circuit("a1")


def a7_target() -> QuantumCircuit:
    return _fetch_circuit("a7")


def a4_target() -> QuantumCircuit:
    return _fetch_circuit("a4")