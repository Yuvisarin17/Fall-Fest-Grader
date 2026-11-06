"""Beginner ladder #1-#14. Pass/fail, unlimited attempts, hints phrased as
measurement results rather than amplitudes."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from . import assets
from .core import (Spec, SubmissionError, count_named, cx_count, cx_edges,
                   equiv_up_to_output_perm, fmt_probs, outcome_probs, parse,
                   placeholder_qubits, reduced_fidelity_dm, same_state, split_final_measurements, state_of,
                   substitute, unitary_of)


def ok(msg: str, **extra) -> dict:
    return {"passed": True, "message": msg, **extra}


def no(msg: str, **extra) -> dict:
    return {"passed": False, "message": msg, **extra}


def p(probs: dict, key: str) -> float:
    return probs.get(key, 0.0)


def one_circuit(data, spec: Spec) -> QuantumCircuit:
    if isinstance(data, list):
        if len(data) != 1:
            raise SubmissionError("This challenge takes one circuit.")
        data = data[0]
    return parse(data, spec)


# ---------- Part A ----------

def b1(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=1))
    pr = outcome_probs(qc)
    if p(pr, "1") > 1 - 1e-6:
        return ok("Every shot reads `1`. You just flipped your first qubit.")
    if p(pr, "0") > 1 - 1e-6:
        return no("You got 100% `0`. Every qubit starts as `0`. Which gate flips it?")
    return no(f"You got {fmt_probs(pr)}. That's a coin flip, not a flip. "
              "H makes a coin; you want `1` every single time.")


def b2(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=1))
    pr = outcome_probs(qc)
    p0 = p(pr, "0")
    if abs(p0 - 0.5) < 0.01:
        return ok("50/50. That's superposition: the qubit isn't secretly 0 or 1 until you measure.")
    if p0 > 1 - 1e-6 or p0 < 1e-6:
        return no(f"You got {fmt_probs(pr)}, the same answer every time. "
                  "You need a gate that puts the qubit halfway between `0` and `1`. Try H.")
    return no(f"You got {fmt_probs(pr)}. Close, but it should be 50% `0` and 50% `1`.")


def b3(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=1))
    n_h = count_named(qc, "h")
    pr = outcome_probs(qc)
    if p(pr, "0") > 1 - 1e-6 and n_h >= 2:
        return ok("Back to `0` every time. Two coin flips cancelled out: that's interference.")
    if n_h < 2:
        return no(f"You used {n_h} H gate(s). The point of this one is to use at least two, "
                  "and still end at `0` every time.")
    return no(f"You got {fmt_probs(pr)}. One H makes a 50/50 coin. "
              "What happens if you apply H to that coin again, right away?")


def b4(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=1))
    pr = outcome_probs(qc)
    p0 = p(pr, "0")
    if abs(p0 - 0.75) <= 0.01:
        return ok("75/25. You turned the dial instead of flipping a switch.")
    if abs(p0 - 0.5) < 1e-6:
        return no("You got 50/50. H only does halfway. You need a gate with a dial: "
                  "try `qc.ry(theta, 0)` and find the theta that gives 75% `0`.")
    if p0 < 0.75:
        return no(f"You got {fmt_probs(pr)}. Too much `1`. Turn the angle down.")
    return no(f"You got {fmt_probs(pr)}. Not enough `1` yet. Turn the angle up.")


def _pair_check(pr, want: tuple[str, str], label: str):
    a, b = want
    if len(pr) == 1:
        only = next(iter(pr))
        return (f"You got `{only}` every time. Nothing is random yet: start with a coin (H) on one qubit, "
                "then link the other qubit to it.")
    other = [k for k in pr if k not in want]
    if not other and abs(p(pr, a) - 0.5) < 0.01:
        return None
    if other and all(abs(v - 0.25) < 0.01 for v in pr.values()) and len(pr) == 4:
        return (f"You got {fmt_probs(pr)}: two independent coins. They need to be linked. "
                "Look at `cx` (CNOT): it flips one qubit depending on the other.")
    if other:
        return f"You got {fmt_probs(pr)}. The coins should only ever {label}: only `{a}` or `{b}`."
    return f"You got {fmt_probs(pr)}. Right outcomes, but it should be 50/50 between them."


def _agree_check(data, want, label, other_label):
    qc = one_circuit(data, Spec(max_qubits=2, min_qubits=2))
    pr = outcome_probs(qc)
    if count_named(qc, "cx") == 0:
        return no("This one is about the CNOT gate: use `qc.cx(control, target)` at least once.")
    if set(pr) <= set(want):
        if len(pr) == 1:
            return ok(f"Always `{next(iter(pr))}`: the qubits always {label}. The CNOT tied the target to the control.")
        return ok(f"Only `{want[0]}` or `{want[1]}`, 50/50, and the qubits always {label}. You've jumped ahead: "
                  "that's entanglement, coming up in Part B.")
    if set(pr) <= set(other_label):
        flip = "disagree" if label == "agree" else "agree"
        return no(f"You got {fmt_probs(pr)}: your qubits always {flip}. One extra X on one qubit fixes it.")
    return no(f"You got {fmt_probs(pr)}. Every outcome should be `{want[0]}` or `{want[1]}`. "
              "Remember: CNOT flips the target only when the control is `1`.")


def b5(data, ctx):
    return _agree_check(data, ("00", "11"), "agree", ("01", "10"))


def b6(data, ctx):
    return _agree_check(data, ("01", "10"), "disagree", ("00", "11"))


# ---------- Part B ----------

BELL = {
    "Φ+ (|00⟩+|11⟩)": Statevector([1, 0, 0, 1]) / np.sqrt(2),
    "Φ− (|00⟩−|11⟩)": Statevector([1, 0, 0, -1]) / np.sqrt(2),
    "Ψ+ (|01⟩+|10⟩)": Statevector([0, 1, 1, 0]) / np.sqrt(2),
    "Ψ− (|01⟩−|10⟩)": Statevector([0, 1, -1, 0]) / np.sqrt(2),
}


def b7(data, ctx):
    if not isinstance(data, list) or len(data) != 4:
        raise SubmissionError("Submit a list of four circuits, in the order Φ+, Φ−, Ψ+, Ψ−.")
    problems = []
    for i, ((label, target), text) in enumerate(zip(BELL.items(), data), start=1):
        qc = parse(text, Spec(max_qubits=2, min_qubits=2))
        sv = state_of(qc)
        if same_state(sv, target):
            continue
        if np.allclose(sv.probabilities(), target.probabilities(), atol=1e-6):
            problems.append(f"Circuit {i} ({label}): right odds, wrong sign. The minus sign is invisible "
                            "to a plain measurement. Which gate adds a minus sign to `1`?")
        else:
            pr = outcome_probs(qc)
            problems.append(f"Circuit {i} ({label}): you got {fmt_probs(pr)}.")
    if problems:
        return no(" ".join(problems))
    return ok("All four Bell states. Same odds, different signs: that's phase.")


def b8(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=5, min_qubits=5))
    vec = np.zeros(32, dtype=complex); vec[0] = vec[31] = 1 / np.sqrt(2)
    target = Statevector(vec)
    sv = state_of(qc)
    if not same_state(sv, target):
        if np.allclose(sv.probabilities(), target.probabilities(), atol=1e-6):
            return no("Right odds (50% `00000`, 50% `11111`) but a minus sign crept in. Check for stray Z/S gates.")
        return no(f"You got {fmt_probs(outcome_probs(qc))}. Target: 50% `00000`, 50% `11111`.")
    body, _ = split_final_measurements(qc)
    depth = body.depth()
    if depth <= 4:
        return ok(f"GHZ state, depth {depth}. You spread the entanglement like a tree, not a chain.",
                  badge="fan-out")
    return ok(f"GHZ state, depth {depth}. Passed. Bonus badge if you can get depth to 4 or less: "
              "after the first CNOT, two qubits are already entangled. Can both pass it on at once?")


def b9(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=1))
    plus = QuantumCircuit(1); plus.h(0)
    minus = QuantumCircuit(1); minus.x(0); minus.h(0)
    pp = outcome_probs(qc, prep=plus)
    pm = outcome_probs(qc, prep=minus)
    good_p = p(pp, "0") > 0.99
    good_m = p(pm, "1") > 0.99
    if good_p and good_m:
        return ok("You can tell + from −. Measuring 'in a different basis' is just a gate before the measurement.")
    if abs(p(pp, "0") - 0.5) < 0.02 and abs(p(pm, "0") - 0.5) < 0.02:
        return no("Both inputs give 50/50: measured directly, + and − look identical. "
                  "What gate turns + and − back into `0` and `1`?")
    return no(f"With + you got {fmt_probs(pp)}; with − you got {fmt_probs(pm)}. "
              "Target: + gives `0`, − gives `1`.")


def b10(data, ctx):
    spec = Spec(max_qubits=5, min_qubits=5, placeholders={"oracle": 5})
    qc = one_circuit(data, spec)
    uses = count_named(qc, "oracle")
    if uses == 0:
        return no("Your circuit never calls the oracle. Add it with `qc.append(bv_oracle(...), range(5))`.")
    if uses > 1:
        return no(f"You queried the oracle {uses} times. The trick of this algorithm is that one query is enough.")
    rng = np.random.default_rng()
    secrets = ["0000", "1111", "1010"] + ["".join(rng.choice(["0", "1"], 4)) for _ in range(3)]
    for s in secrets:
        sub = substitute(qc, "oracle", assets.bv_oracle(s))
        pr = outcome_probs(sub, implicit_qubits=[0, 1, 2, 3])
        best = max(pr, key=pr.get)
        if len(best) != 4:
            return no(f"Measure exactly the 4 input qubits (0 to 3) into 4 classical bits. "
                      f"Your output had {len(best)} bits.")
        if pr[best] < 0.99 or best != s:
            return no(f"With a hidden secret of `{s}`, you got {fmt_probs(pr)}. "
                      "Hints: put the ancilla (qubit 4) in the − state before the oracle, "
                      "and sandwich the input qubits in H gates.")
    return ok("You found every secret with one query. A classical computer needs 4.")


def b11(data, ctx):
    spec = Spec(max_qubits=2, min_qubits=2, placeholders={"oracle": 2})
    qc = one_circuit(data, spec)
    if count_named(qc, "oracle") == 0:
        return no("Your circuit never calls the oracle.")
    for m in ["00", "01", "10", "11"]:
        pr = outcome_probs(substitute(qc, "oracle", assets.grover_oracle(m)))
        if p(pr, m) < 0.9:
            return no(f"When the marked card is `{m}`, you got {fmt_probs(pr)}. "
                      "After the oracle, you need the 'diffuser' (reflect about the average) to boost the marked card.")
    return ok("You found the marked card every time, in one oracle call.")


def _random_states(rng, k):
    plus = QuantumCircuit(1); plus.h(0)
    plus_i = QuantumCircuit(1); plus_i.h(0); plus_i.s(0)
    minus = QuantumCircuit(1); minus.x(0); minus.h(0)
    out = [plus, plus_i, minus]
    for _ in range(k):
        c = QuantumCircuit(1)
        c.u(*rng.uniform(0, 2 * np.pi, 3), 0)
        out.append(c)
    return out


def _dm_fidelity_runs(circ_for_prep, preps, qubit, shots=512, noise_model=None):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    fids = []
    for prep, target in preps:
        c = circ_for_prep(prep)
        c.save_density_matrix(qubits=[qubit], label="rho")
        res = sim.run(c, shots=shots, seed_simulator=1234).result().data()["rho"]
        fids.append(reduced_fidelity_dm(res, target))
    return fids


def _strip_trailing_measure_on(qc: QuantumCircuit, q: int) -> QuantumCircuit:
    """Remove measurements on qubit q (so we can inspect its state)."""
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "measure" and qc.find_bit(inst.qubits[0]).index == q:
            continue
        out.append(inst)
    return out


def b12(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=3, min_qubits=3, allow_dynamic=True))
    # no shortcuts: nothing may touch qubits 0 and 2 together
    for inst in qc.data:
        qs = {qc.find_bit(q).index for q in inst.qubits}
        if inst.operation.name not in ("barrier", "if_else") and {0, 2} <= qs:
            return no("No gates directly between qubit 0 and qubit 2: the whole point is that they never touch.")
    if count_named(qc, "if_else") == 0:
        return no("Teleportation needs corrections that depend on measurement results. Use `with qc.if_test(...)`.")
    body = _strip_trailing_measure_on(qc, 2)
    rng = np.random.default_rng()
    preps = [(pr, Statevector(pr)) for pr in _random_states(rng, 2)]

    def build(prep):
        c = body.copy_empty_like()
        c.compose(prep, qubits=[0], inplace=True)
        c.compose(body, inplace=True)
        return c

    fids = _dm_fidelity_runs(build, preps, qubit=2)
    worst = min(fids)
    if worst > 0.99:
        return ok("Teleported. Qubit 2 ends up in the input state without ever interacting with qubit 0.")
    return no(f"Qubit 2 matched the input only {worst*100:.0f}% of the time on some inputs. "
              "Check both corrections: X if one measurement is 1, Z if the other is 1.")


def b13(data, ctx):
    spec = Spec(max_qubits=5, min_qubits=3, allow_dynamic=True, placeholders={"noise": 3})
    qc = one_circuit(data, spec)
    locs = placeholder_qubits(qc, "noise")
    if count_named(qc, "noise") != 1 or len(locs) != 1:
        return no("Use the `noise` gate exactly once, outside any if_test.")
    if locs[0] != [0, 1, 2]:
        return no("Apply the `noise` gate to qubits 0, 1, 2 (your three code qubits), in that order.")
    rng = np.random.default_rng()
    states = _random_states(rng, 1)
    body = _strip_trailing_measure_on(qc, 0)
    for k in (None, 0, 1, 2):
        variant = substitute(body, "noise", assets.bitflip(k))

        def build(prep, variant=variant):
            c = variant.copy_empty_like()
            c.compose(prep, qubits=[0], inplace=True)
            c.compose(variant, inplace=True)
            return c

        fids = _dm_fidelity_runs(build, [(s, Statevector(s)) for s in states], qubit=0, shots=256)
        if min(fids) < 0.99:
            where = "with no error" if k is None else f"when qubit {k} gets flipped"
            return no(f"Qubit 0 didn't get its original state back {where} "
                      f"(match: {max(0.0, min(fids))*100:.0f}%). Encode with 2 CNOTs, measure the parity of pairs, "
                      "fix the qubit the parities point to, then decode.")
    return ok("Every single bit-flip got caught and fixed. You just ran a quantum error-correcting code.")


def b14(data, ctx):
    qc = one_circuit(data, Spec(max_qubits=4, min_qubits=4))
    bad = [e for e in cx_edges(qc) if e not in assets.LINE4]
    if bad:
        a, b = bad[0]
        return no(f"Qubits {a} and {b} aren't neighbors on this chip (0-1-2-3). Move one next to the other with SWAPs first.")
    target = assets.b14_target()
    perm = equiv_up_to_output_perm(unitary_of(qc), unitary_of(target), 4)
    if perm is None:
        return no("Your circuit doesn't do the same thing as the original. Every gate of the original "
                  "must still happen, on whichever physical qubit now holds that logical qubit.")
    n = cx_count(qc)
    budget = 3 + 2 * 3
    if n > budget:
        return no(f"Your circuit uses {n} CNOTs (a SWAP costs 3). Budget is {budget}: the 3 original "
                  "CNOTs plus at most 2 SWAPs.")
    moved = "" if perm == (0, 1, 2, 3) else f" Your qubits ended up shuffled ({perm}), which is fine: real compilers track that too."
    return ok(f"Fits on the chip with {n} CNOTs.{moved}")


CHECKS = {f"b{i}": f for i, f in enumerate([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14], start=1)}