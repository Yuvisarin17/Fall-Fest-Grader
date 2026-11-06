"""Safe parsing and simulation helpers shared by every challenge.

Submissions arrive as OpenQASM text only. Nothing a participant sends is ever
executed as Python: we parse QASM, check it against an allowlist, enforce size
limits, and only then simulate.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit.circuit import ControlFlowOp, Gate
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

MAX_QASM_CHARS = 60_000
MAX_EXPANDED_OPS = 4_000

ONE_Q = {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
         "rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "r", "U"}
TWO_Q = {"cx", "cz", "cy", "ch", "swap", "crx", "cry", "crz", "cp", "cu", "cu1", "cu3",
         "rxx", "ryy", "rzz", "rzx", "ecr", "iswap", "csx", "dcx", "CX"}
THREE_Q = {"ccx", "cswap", "ccz"}
STRUCTURAL = {"measure", "barrier", "reset", "delay"}
CONTROL_FLOW = {"if_else"}  # no loops: keeps runtime bounded
STANDARD = ONE_Q | TWO_Q | THREE_Q


class SubmissionError(Exception):
    """Problem with what the participant sent. The message is shown to them."""


@dataclass
class Spec:
    max_qubits: int
    min_qubits: int = 1
    allow_dynamic: bool = False          # mid-circuit measurement / if_test
    placeholders: dict = field(default_factory=dict)  # name -> num_qubits


def parse(text: str, spec: Spec) -> QuantumCircuit:
    if not isinstance(text, str):
        raise SubmissionError("Expected a QuantumCircuit.")
    if len(text) > MAX_QASM_CHARS:
        raise SubmissionError(f"Circuit is too large ({len(text)} characters).")
    head = text.lstrip()[:40]
    try:
        if head.startswith("OPENQASM 2"):
            qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        elif head.startswith("OPENQASM 3"):
            qc = qasm3.loads(text)
        else:
            raise SubmissionError("Expected a QuantumCircuit (or OpenQASM text).")
    except SubmissionError:
        raise
    except Exception as e:  # parser errors
        raise SubmissionError(f"Could not read your circuit: {str(e)[:300]}") from None

    if qc.num_qubits > spec.max_qubits:
        raise SubmissionError(f"This challenge allows at most {spec.max_qubits} qubits; you used {qc.num_qubits}.")
    if qc.num_qubits < spec.min_qubits:
        raise SubmissionError(f"This challenge needs at least {spec.min_qubits} qubits; you used {qc.num_qubits}.")
    qc = _flatten(qc, spec)
    return qc


def _flatten(qc: QuantumCircuit, spec: Spec) -> QuantumCircuit:
    """Expand user-defined gates, enforce the allowlist and the op budget."""
    budget = [MAX_EXPANDED_OPS]

    def walk(circ: QuantumCircuit, top: bool) -> QuantumCircuit:
        out = circ.copy_empty_like()
        for inst in circ.data:
            op = inst.operation
            name = op.name
            budget[0] -= 1
            if budget[0] < 0:
                raise SubmissionError("Circuit is too long after expanding custom gates.")
            if name in spec.placeholders:
                if op.num_qubits != spec.placeholders[name]:
                    raise SubmissionError(f"The '{name}' gate must act on {spec.placeholders[name]} qubits.")
                out.append(Gate(name, op.num_qubits, []), inst.qubits, inst.clbits)
            elif isinstance(op, ControlFlowOp):
                if not spec.allow_dynamic:
                    raise SubmissionError("This challenge doesn't use if_test or other classical control.")
                if name not in CONTROL_FLOW:
                    raise SubmissionError(f"'{name}' is not allowed. Only if_test / if-else is supported.")
                blocks = [walk(b, False) for b in op.blocks]
                out.append(op.replace_blocks(blocks), inst.qubits, inst.clbits)
            elif name in STANDARD or name in STRUCTURAL:
                if any(not _is_number(p) for p in op.params):
                    raise SubmissionError(f"Gate '{name}' has unbound parameters. Bind numbers before submitting.")
                out.append(op, inst.qubits, inst.clbits)
            elif op.definition is not None and isinstance(op, Gate):
                sub = walk(op.definition, False)
                out.compose(sub, qubits=inst.qubits, clbits=inst.clbits, inplace=True)
            else:
                raise SubmissionError(f"Gate '{name}' is not allowed in this challenge.")
        return out

    flat = walk(qc, True)
    if not spec.allow_dynamic and _has_mid_measure(flat):
        raise SubmissionError("Measurements must come at the end of the circuit for this challenge.")
    return flat


def _is_number(p) -> bool:
    try:
        float(p)
        return True
    except Exception:
        return False


def _has_mid_measure(qc: QuantumCircuit) -> bool:
    measured = set()
    for inst in qc.data:
        name = inst.operation.name
        qs = {qc.find_bit(q).index for q in inst.qubits}
        if name == "measure":
            measured |= qs
        elif name not in ("barrier",) and qs & measured:
            return True
    return False


# ---------- outcome helpers ----------

def split_final_measurements(qc: QuantumCircuit):
    """Return (quantum-only circuit, {clbit_index: qubit_index}) for circuits whose
    measurements are all at the end. Barriers and measurements are dropped."""
    body = QuantumCircuit(qc.num_qubits)
    cmap = {}
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "measure":
            cmap[qc.find_bit(inst.clbits[0]).index] = qs[0]
        elif name in ("barrier", "delay"):
            continue
        else:
            body.append(inst.operation, qs)
    return body, cmap


def state_of(qc: QuantumCircuit, prep: QuantumCircuit | None = None) -> Statevector:
    body, _ = split_final_measurements(qc)
    if prep is not None:
        body = prep.compose(body)
    return Statevector(body)


def outcome_probs(qc: QuantumCircuit, prep: QuantumCircuit | None = None,
                  implicit_qubits: list[int] | None = None) -> dict[str, float]:
    """Exact measurement distribution, keyed like Qiskit counts (c[n-1]...c[0]).
    If the circuit has no measurements, the qubits in implicit_qubits (default all)
    are measured into clbits 0..k-1."""
    body, cmap = split_final_measurements(qc)
    if not cmap:
        qs = implicit_qubits if implicit_qubits is not None else list(range(qc.num_qubits))
        cmap = {i: q for i, q in enumerate(qs)}
    sv = state_of(body, prep)
    nclb = max(cmap) + 1
    order = [cmap.get(c) for c in range(nclb)]
    qubits = [q for q in order if q is not None]
    probs = sv.probabilities_dict(qargs=qubits)
    out: dict[str, float] = {}
    for key, p in probs.items():
        # key is ordered qubits[-1] ... qubits[0]
        bits = dict(zip(reversed(qubits), key))
        s = "".join(bits[cmap[c]] if c in cmap else "0" for c in reversed(range(nclb)))
        out[s] = out.get(s, 0.0) + float(p)
    return {k: v for k, v in out.items() if v > 1e-9}


def fmt_probs(p: dict[str, float], top: int = 4) -> str:
    items = sorted(p.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{v*100:.0f}% `{k}`" for k, v in items)


def same_state(a: Statevector, b: Statevector, tol: float = 1e-6) -> bool:
    return abs(abs(np.vdot(a.data, b.data)) - 1) < tol


def substitute(qc: QuantumCircuit, name: str, replacement: QuantumCircuit) -> QuantumCircuit:
    out = qc.copy_empty_like()
    for inst in qc.data:
        op = inst.operation
        if op.name == name:
            out.compose(replacement, qubits=inst.qubits, inplace=True)
        elif isinstance(op, ControlFlowOp):
            out.append(op.replace_blocks([substitute(b, name, replacement) for b in op.blocks]),
                       inst.qubits, inst.clbits)
        else:
            out.append(inst)
    return out


def count_named(qc: QuantumCircuit, name: str) -> int:
    n = 0
    for inst in qc.data:
        if inst.operation.name == name:
            n += 1
        elif isinstance(inst.operation, ControlFlowOp):
            n += sum(count_named(b, name) for b in inst.operation.blocks)
    return n


def placeholder_qubits(qc: QuantumCircuit, name: str) -> list[list[int]]:
    return [[qc.find_bit(q).index for q in inst.qubits] for inst in qc.data if inst.operation.name == name]


def to_cx_basis(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose to CX + single-qubit gates without optimizing, so the count
    reflects what the participant actually wrote (a SWAP costs 3, a CCX costs 6)."""
    body, _ = split_final_measurements(qc)
    return transpile(body, basis_gates=["cx", "u"], optimization_level=0)


def cx_count(qc: QuantumCircuit) -> int:
    return to_cx_basis(qc).count_ops().get("cx", 0)


def cx_edges(qc: QuantumCircuit) -> set[tuple[int, int]]:
    c = to_cx_basis(qc)
    return {tuple(sorted(c.find_bit(q).index for q in inst.qubits))
            for inst in c.data if inst.operation.name == "cx"}


def unitary_of(qc: QuantumCircuit) -> np.ndarray:
    body, _ = split_final_measurements(qc)
    return Operator(body).data


def equiv_up_to_output_perm(u_sub: np.ndarray, u_target: np.ndarray, n: int):
    """Find a qubit permutation P with U_sub == P @ U_target (global phase ignored).
    Returns the permutation (tuple) or None."""
    dim = 2 ** n
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(n)) & 1
    for perm in itertools.permutations(range(n)):
        # output qubit perm[i] holds logical qubit i
        newidx = (bits[:, :] << np.array(perm)).sum(axis=1)
        permuted = np.empty_like(u_target)
        permuted[newidx, :] = u_target
        ov = abs(np.vdot(permuted, u_sub)) / dim
        if abs(ov - 1) < 1e-6:
            return perm
    return None


def reduced_fidelity_dm(dm, target: Statevector) -> float:
    return float(np.real(DensityMatrix(dm).expectation_value(target.to_operator())))
