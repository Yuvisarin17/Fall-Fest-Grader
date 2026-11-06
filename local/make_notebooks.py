"""Generate notebooks/beginner.ipynb and notebooks/advanced.ipynb.
Set GRADER_URL to wherever dist/fallfest.py is hosted (e.g. a raw GitHub URL), then:
    python local/make_notebooks.py
"""
import pathlib
import sys

import nbformat as nbf

GRADER_URL = sys.argv[1] if len(sys.argv) > 1 else "https://raw.githubusercontent.com/YOUR-ORG/fallfest-2026/main/fallfest.py"
FORM_URL = sys.argv[2] if len(sys.argv) > 2 else "https://forms.gle/YOUR-FORM"
ROOT = pathlib.Path(__file__).resolve().parents[1]

SETUP = f'''# Run this cell first (takes about a minute the first time)
%pip install -q qiskit qiskit-aer qiskit-qasm3-import
import os, urllib.request
if not os.path.exists("fallfest.py"):
    urllib.request.urlretrieve("{GRADER_URL}", "fallfest.py")
from fallfest import *'''

CONCEPT = "> 📖 **Concept:** _organizers: replace this line with your description._"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def challenge(cid, title, task, scaffold, extra_md=""):
    return [md(f"## {cid.upper()}: {title}\n\n{CONCEPT}\n\n**Your task:** {task}\n\n{extra_md}"), code(scaffold)]


def beginner():
    cells = [
        md(f"""
# Qiskit Fall Fest 2026: Beginner Ladder

Work down the list in order. Each challenge adds one new idea.

1. Run the setup cell below.
2. In each challenge, write your gates where it says `# YOUR CODE`.
3. Run the cell. `check(...)` tells you right away whether you got it, with a hint if not.
4. When you pass, copy the code it prints into the **[submission form]({FORM_URL})** so it counts for your team.

`counts(qc)` runs your circuit 1000 times and shows what came out. Use it whenever you're unsure.
"""),
        code(SETUP),
        md("# Part A: No quantum background needed"),
    ]
    cells += challenge("b1", "Flip it", "make the qubit measure `1` every time. Every qubit starts as `0`.", """
qc = QuantumCircuit(1, 1)
# YOUR CODE: one gate goes here

qc.measure(0, 0)
print(counts(qc))
check("b1", qc)
""")
    cells += challenge("b2", "Quantum coin", "make the qubit measure `0` half the time and `1` half the time.", """
qc = QuantumCircuit(1, 1)
# YOUR CODE

qc.measure(0, 0)
print(counts(qc))
check("b2", qc)
""")
    cells += challenge("b3", "Undo the magic", "use **at least two** `h` gates and still end at `0` every time.", """
qc = QuantumCircuit(1, 1)
# YOUR CODE

qc.measure(0, 0)
print(counts(qc))
check("b3", qc)
""")
    cells += challenge("b4", "Loaded coin", "a coin that lands `0` 75% of the time and `1` 25% of the time.", """
import numpy as np
theta = 0   # YOUR CODE: find the right angle (np.pi is available)

qc = QuantumCircuit(1, 1)
qc.ry(theta, 0)
qc.measure(0, 0)
print(counts(qc))
check("b4", qc)
""", "`qc.ry(theta, 0)` is a dial instead of a switch: `theta = 0` does nothing, `theta = np.pi` is a full flip.")
    cells += challenge("b5", "Coins that agree", "two qubits that only ever give `00` or `11`, 50/50.", """
qc = QuantumCircuit(2, 2)
# YOUR CODE: you'll need qc.cx(control, target)

qc.measure([0, 1], [0, 1])
print(counts(qc))
check("b5", qc)
""", "`qc.cx(0, 1)` flips qubit 1 only when qubit 0 is `1`.")
    cells += challenge("b6", "Coins that disagree", "two qubits that only ever give `01` or `10`, 50/50.", """
qc = QuantumCircuit(2, 2)
# YOUR CODE

qc.measure([0, 1], [0, 1])
print(counts(qc))
check("b6", qc)
""")
    cells.append(md("# Part B: Beginner"))
    cells += challenge("b7", "Bell zoo", "build all four Bell states, in this order: "
                       "Φ+ = (|00⟩+|11⟩)/√2, Φ− = (|00⟩−|11⟩)/√2, Ψ+ = (|01⟩+|10⟩)/√2, Ψ− = (|01⟩−|10⟩)/√2. "
                       "No measurements needed.", """
phi_plus = QuantumCircuit(2)
# YOUR CODE

phi_minus = QuantumCircuit(2)
# YOUR CODE

psi_plus = QuantumCircuit(2)
# YOUR CODE

psi_minus = QuantumCircuit(2)
# YOUR CODE

check("b7", [phi_plus, phi_minus, psi_plus, psi_minus])
""", "Φ+ and Φ− give identical counts. The difference is a minus sign that only a gate like `z` can add.")
    cells += challenge("b8", "GHZ fan-out", "5 qubits that give only `00000` or `11111`, 50/50. "
                       "Bonus badge if your circuit depth is 4 or less.", """
qc = QuantumCircuit(5)
# YOUR CODE

print("depth:", qc.depth())
print(counts(qc))
check("b8", qc)
""")
    cells += challenge("b9", "Plus or minus?", "the grader puts qubit 0 in either |+⟩ or |−⟩ before your circuit runs. "
                       "Your circuit must output `0` for |+⟩ and `1` for |−⟩.", """
qc = QuantumCircuit(1, 1)
# YOUR CODE

qc.measure(0, 0)

# Try it yourself: prepare |+> or |->, then run your circuit
plus = QuantumCircuit(1, 1); plus.h(0)
minus = QuantumCircuit(1, 1); minus.x(0); minus.h(0)
print("+ ->", counts(plus.compose(qc)), "   - ->", counts(minus.compose(qc)))
check("b9", qc)
""")
    cells += challenge("b10", "Find the secret number", "the oracle hides a 4-bit secret. Find it by calling the oracle **once**. "
                       "Qubits 0 to 3 are inputs, qubit 4 is a helper. Measure qubits 0 to 3.", """
qc = QuantumCircuit(5, 4)
# YOUR CODE: before the oracle

qc.append(bv_oracle("1011"), range(5))   # the grader swaps in random secrets
# YOUR CODE: after the oracle

qc.measure(range(4), range(4))
print(counts(qc))   # should show 100% 1011
check("b10", qc)
""")
    cells += challenge("b11", "Find the marked card", "4 cards (`00`, `01`, `10`, `11`), one is marked. "
                       "Call the oracle once and measure the marked card at least 90% of the time.", """
qc = QuantumCircuit(2, 2)
# YOUR CODE: before the oracle

qc.append(grover_oracle("10"), [0, 1])   # the grader tests all four cards
# YOUR CODE: the diffuser

qc.measure([0, 1], [0, 1])
print(counts(qc))   # should show 100% 10
check("b11", qc)
""")
    cells += challenge("b12", "Teleport it", "the grader puts a random state on qubit 0. Move it to qubit 2. "
                       "Qubits 0 and 2 may never share a gate. Use `qc.if_test` for the corrections.", """
qc = QuantumCircuit(3, 2)
# YOUR CODE: entangle qubits 1 and 2, then the Bell measurement on 0 and 1

qc.measure(0, 0)
qc.measure(1, 1)
with qc.if_test((qc.clbits[1], 1)):
    pass   # YOUR CODE: correction on qubit 2
with qc.if_test((qc.clbits[0], 1)):
    pass   # YOUR CODE: correction on qubit 2

check("b12", qc)
""")
    cells += challenge("b13", "Fix the flip", "protect qubit 0 with a 3-qubit repetition code. The `noise_gate` may flip "
                       "any one of qubits 0, 1, 2. Afterwards, qubit 0 must be back in its original state. "
                       "Qubits 3 and 4 are free to use as helpers.", """
qc = QuantumCircuit(5, 2)
# YOUR CODE: encode qubit 0 into qubits 0, 1, 2

qc.append(noise_gate(flip=1), [0, 1, 2])   # the grader tries every single flip
# YOUR CODE: find which qubit flipped (helpers 3 and 4), fix it, then decode

check("b13", qc)
""")
    cells += challenge("b14", "Fit on the chip", "on this chip, qubits are in a line (0-1-2-3) and a CNOT only works "
                       "between neighbors. Rewrite the circuit below so it runs, using at most 2 SWAPs.", """
target = b14_target()
print(target.draw())

qc = QuantumCircuit(4)
# YOUR CODE: same circuit, neighbor-only CNOTs, qc.swap(a, b) allowed

check("b14", qc)
""", "Your qubits can end up in a different order. That's allowed.")
    return cells


def advanced():
    cells = [
        md(f"""
# Qiskit Fall Fest 2026: Advanced Challenges

Every challenge is scored. Run `check(...)` as often as you like. When you're happy with a score, paste the
code it prints into the **[submission form]({FORM_URL})**. The organizers re-grade every code, and your best one counts.

Rules: CNOT counts are taken after decomposing to CX + single-qubit gates without optimizing
(a SWAP costs 3, a Toffoli costs 6). Qubit caps and coupling maps are enforced.
"""),
        code(SETUP),
    ]
    cells += challenge("a1", "Circuit golf", "rewrite the target with as few CNOTs as possible, using only CNOTs "
                       "between neighbors on the line 0-1-2-3. A final reordering of qubits is allowed. **Lower is better.**", """
target = a1_target()
print(target.draw(fold=-1))

qc = QuantumCircuit(4)
# YOUR CODE

check("a1", qc)
""")
    cells += challenge("a2", "QAOA MaxCut", f"maximize the expected cut on the fixed 10-node graph `A2_EDGES`, using up to "
                       f"p = 5 QAOA layers. Submit the angles; the grader builds the standard QAOA circuit "
                       f"(H on all qubits, then per layer `rzz(-gamma)` on every edge and `rx(2*beta)` on every qubit). "
                       f"Score: expected cut / max cut. **Higher is better**, ties go to fewer layers.", """
print(A2_EDGES)
gammas = [0.5]   # YOUR CODE: optimize these
betas = [0.4]

check("a2", {"gammas": gammas, "betas": betas})
""")
    cells += challenge("a3", "Ground state hunt", "prepare the ground state of the 6-qubit transverse-field Ising chain "
                       "H = −Σ Z<sub>i</sub>Z<sub>i+1</sub> − Σ X<sub>i</sub> (open chain). Submit the circuit with angles bound. "
                       "Score: (your energy − exact energy) + 0.005 × CNOTs, max 60 CNOTs. **Lower is better.**", """
H = tfim(6)   # SparsePauliOp
qc = QuantumCircuit(6)
# YOUR CODE: ansatz + optimized parameters (bind them before checking)

check("a3", qc)
""")
    cells += challenge("a4", "Beat the noise", "estimate ⟨ZZZZ⟩ for `a4_target()` using only `run_noisy(...)`, which runs "
                       "circuits under a fixed noise model (gate errors + readout errors). Raw results are badly biased; "
                       "use error mitigation. Submit one number. Scores are revealed at the end. **Closest to the true value wins.**",
                       """
target = a4_target()
meas = target.copy(); meas.measure_all()
raw = run_noisy(meas, shots=20000)[0]
raw_estimate = sum((-1) ** k.count("1") * v for k, v in raw.items()) / 20000
print("raw:", raw_estimate)

estimate = raw_estimate   # YOUR CODE: mitigate
check("a4", estimate)
""", "**Rule:** computing the answer with an exact or noiseless simulator is not allowed. Top teams will be asked to walk through their method.")
    cells += challenge("a5", "Trotter tradeoff", "starting from |00000⟩, approximate e<sup>−iHt</sup>|00000⟩ for the 5-qubit "
                       "version of the A3 Hamiltonian at t = 1.5. Score: fidelity − 0.002 × CNOTs. **Higher is better.**", """
H = tfim(5)
qc = QuantumCircuit(5)
# YOUR CODE: Trotter steps (qc.rzz and qc.rx are handy)

check("a5", qc)
""")
    cells += challenge("a6", "Beat the physical error rate", "protect qubit 0 through `idle_gate()`, which flips each of "
                       "qubits 0, 1, 2 with 10% probability. Your own gates are noisy too. Up to 5 qubits, mid-circuit "
                       "measurement allowed. Score: logical error averaged over 6 input states; it must beat a bare "
                       "qubit (0.067) to count. **Lower is better.**", """
qc = QuantumCircuit(5, 2)
# YOUR CODE: encode

qc.append(idle_gate(), [0, 1, 2])
# YOUR CODE: decode / correct

check("a6", qc)
""")
    cells += challenge("a7", "Route it", "make the 6-qubit target legal on the heavy-hex fragment `HEAVYHEX6` "
                       "(edges 0-1, 1-2, 2-3, 3-4, 2-5) with as few CNOTs as possible. Logical qubit i starts on physical "
                       "qubit i; a final reordering is allowed. Qiskit's best effort is 29. **Lower is better.**", """
target = a7_target()
print(HEAVYHEX6)

qc = QuantumCircuit(6)
# YOUR CODE

check("a7", qc)
""")
    return cells


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out)


write("beginner.ipynb", beginner())
write("advanced.ipynb", advanced())
