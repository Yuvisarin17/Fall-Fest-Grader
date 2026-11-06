# ---- participant-facing API (appended to the bundled grader) ----
import base64 as _b64
import json as _json
import zlib as _zlib

from qiskit import QuantumCircuit, qasm2 as _qasm2, qasm3 as _qasm3
from qiskit.circuit import ControlFlowOp as _CF

from _ffgrader import assets as _assets
from _ffgrader import registry as _registry
from _ffgrader.advanced import a4_run as _a4_run

A4_SEED = 2026
SHOW_CODES = True

__all__ = ["QuantumCircuit", "check", "counts", "run_noisy", "submission_code",
           "bv_oracle", "grover_oracle", "noise_gate", "idle_gate",
           "b14_target", "a1_target", "a7_target", "a4_target",
           "A2_EDGES", "A2_MAX_P", "LINE4", "HEAVYHEX6", "tfim", "info"]

A2_EDGES = _assets.A2_EDGES
A2_MAX_P = _assets.A2_MAX_P
LINE4 = sorted(_assets.LINE4)
HEAVYHEX6 = sorted(_assets.HEAVYHEX6)
tfim = _assets.tfim


def to_qasm(qc):
    if not isinstance(qc, QuantumCircuit):
        raise TypeError("Expected a QuantumCircuit.")
    if qc.num_parameters:
        raise ValueError("Your circuit still has unbound parameters. Use qc.assign_parameters(...) first.")
    if any(isinstance(i.operation, _CF) for i in qc.data):
        return _qasm3.dumps(qc)
    try:
        return _qasm2.dumps(qc)
    except Exception:
        return _qasm3.dumps(qc)


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


def submission_code(challenge, answer) -> str:
    """A short code to paste into the submission form. Organizers re-grade it."""
    raw = _json.dumps({"c": challenge.lower(), "d": _encode(answer)}, separators=(",", ":"))
    return "FF1:" + _b64.urlsafe_b64encode(_zlib.compress(raw.encode(), 9)).decode()


def check(challenge, answer):
    """Grade an answer right here in the notebook and print feedback."""
    cid = challenge.lower().strip()
    if cid not in _registry.META:
        print(f"❌ There's no challenge called '{challenge}'.")
        return None
    try:
        data = _encode(answer)
    except Exception as e:
        print(f"❌ {challenge.upper()}: {e}")
        return None
    res = _registry.grade(cid, data, {"team_seed": A4_SEED})
    if "error" in res:
        print(f"❌ {cid.upper()}: {res['error']}")
        if "internal" in res:
            print(res["internal"])
        return res
    if "passed" in res:
        print(("✅ " if res["passed"] else "❌ ") + f"{cid.upper()}: {res['message']}")
        if res.get("badge"):
            print(f"🏅 Badge earned: {res['badge']}")
        good = res["passed"]
    else:
        if res.get("hidden"):
            print(f"📨 {cid.upper()}: estimate recorded locally. The organizers score this one at the end.")
        elif not res.get("valid", True):
            print(f"❌ {cid.upper()}: {res['message']} (score {res['score']:.6g}, does not count)"
                  if res["score"] != float("inf") else f"❌ {cid.upper()}: {res['message']}")
        else:
            print(f"📈 {cid.upper()} score: {res['score']:.6g}. {res['message']}")
        good = res.get("valid", True)
    if SHOW_CODES and good:
        print("\nPaste this into the submission form:\n" + submission_code(cid, answer))
    return res


def counts(qc, shots=1000):
    """Run a circuit on a perfect simulator and return the measurement counts."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    c = qc.copy()
    if c.num_clbits == 0:
        c.measure_all()
    sim = AerSimulator()
    return dict(sorted(sim.run(transpile(c, sim), shots=shots).result().get_counts().items()))


def run_noisy(circuits, shots=4000):
    """A4: run circuits on the shared noise model. Returns a list of counts dicts."""
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    return _a4_run([to_qasm(c) for c in circuits], shots, {"run_seed": None})


def info(challenge):
    return _registry.info(challenge.lower(), A4_SEED)


def bv_oracle(secret: str):
    """B10: 5-qubit oracle hiding a 4-bit secret (qubits 0-3 inputs, qubit 4 ancilla).
    check() swaps in its own random secrets, so any secret works for testing."""
    return _assets.bv_oracle(secret).to_gate()


def grover_oracle(marked: str):
    """B11: flips the sign of the 2-qubit state |marked>. check() tests all four."""
    return _assets.grover_oracle(marked).to_gate()


def noise_gate(flip=None):
    """B13: put this on qubits [0, 1, 2]. It flips qubit `flip` (or nothing).
    check() tries no flip and a flip on each of the three qubits."""
    return _assets.bitflip(flip).to_gate()


def idle_gate():
    """A6: marks where your encoded qubits sit idle. check() applies a 10% bit flip
    to each of qubits 0, 1, 2 here."""
    return QuantumCircuit(3, name="idle").to_gate()


def b14_target():
    return _assets.b14_target()


def a1_target():
    return _assets.a1_target()


def a7_target():
    return _assets.a7_target()


def a4_target():
    return _assets.a4_target(A4_SEED)


print("Fall Fest grader loaded. Use check('b1', qc) to grade an answer.")
