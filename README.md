# Qiskit Fall Fest 2026 Grader

Participants build circuits locally (Colab is fine) and call `submit("b1", qc)`.
Only OpenQASM text or numbers reach the server; nothing they send is executed as Python.

## Run it

```bash
pip install -r requirements.txt
export ADMIN_TOKEN=pick-a-long-secret
uvicorn server:app --host 0.0.0.0 --port 8000      # one uvicorn process; grading uses its own worker pool
```

Open `http://<host>:8000/` for the leaderboard (auto-refreshes, built for a projector).

Env knobs: `GRADER_WORKERS` (default 2, set to CPU count), `GRADER_TIMEOUT` (40s), `SUBMIT_MIN_GAP` (3s per team), `FALLFEST_DB`.

## Teams

```bash
python manage.py import teams.txt     # one name per line, writes tokens.csv
python manage.py add "Late Team"
python manage.py list
python manage.py freeze on            # close submissions at the end
python manage.py reveal on            # show A4 scores (hidden until then)
```

## Participant setup (first cell of every notebook)

```python
!pip install -q qiskit qiskit-aer requests
!wget -q http://<host>:8000/fallfest.py
from fallfest import *
login("TEAM-TOKEN", server="http://<host>:8000")
```

Helpers: `submit`, `bv_oracle` (B10), `grover_oracle` (B11), `noise_gate` (B13), `idle_gate` (A6),
`b14_target()`, `a1_target()`, `a7_target()`, `a4_target()`, `run_noisy` (A4), `challenge_info`, `my_progress`.

## What each challenge expects

| ID | Submit | Notes |
|---|---|---|
| b1-b6, b9 | one circuit | Measurements optional; with none, all qubits are measured |
| b7 | list of 4 circuits | Order Φ+, Φ−, Ψ+, Ψ− |
| b8 | 5-qubit circuit | Badge `fan-out` if depth ≤ 4 |
| b10 | 5-qubit circuit with `bv_oracle(...)` once | Grader swaps in 6 secrets; measure qubits 0-3 |
| b11 | 2-qubit circuit with `grover_oracle(...)` | Grader tests all 4 marked states |
| b12 | 3-qubit dynamic circuit | Must use `if_test`; no gate may touch qubits 0 and 2 together |
| b13 | ≤5 qubits, `noise_gate()` once on [0,1,2] | Tested with no flip and a flip on each qubit, random inputs |
| b14 | 4-qubit circuit on line 0-1-2-3 | ≤ 9 CNOTs (3 original + 2 SWAPs); final relabelling allowed |
| a1 | 4-qubit circuit on line | Score: CNOTs (lower). Relabelling allowed |
| a2 | `{"gammas": [...], "betas": [...]}` | p ≤ 5. Score: approximation ratio (higher), tiebreak p |
| a3 | 6-qubit circuit | Score: energy error + 0.005·CNOTs (lower), cap 60 CNOTs |
| a4 | a number | Per-team circuit, 400k shot budget via `run_noisy`; score hidden until reveal; last submission counts |
| a5 | 5-qubit circuit | Score: fidelity − 0.002·CNOTs (higher), t = 1.5 |
| a6 | ≤5 qubits, `idle_gate()` once on [0,1,2] | Score: logical error over 6 input states (lower); counts only if it beats a bare qubit (0.067) |
| a7 | 6-qubit circuit on heavy-hex fragment | Score: CNOTs (lower); bonus flag for beating Qiskit opt level 3 (29) |

Reference scores from `tests/`: A1 18, A2 0.872 (p=2), A3 0.087, A5 0.83, A6 0.033, A7 29.

## Test before the event

```bash
python manage.py add T1; python manage.py add T2       # two throwaway tokens
SUBMIT_MIN_GAP=0 uvicorn server:app --port 8000 &
FALLFEST_SERVER=http://localhost:8000 python tests/test_grader.py <T1> <T2>
```

Runs every reference solution, common wrong answers, and abuse cases (oversized circuits,
exponential gate nesting, non-QASM text, bad tokens, shot budget). Use a fresh DB for the event.

## Safety limits

QASM only (2 or 3), 60k characters, 4,000 operations after expanding custom gates, per-challenge qubit caps,
gate allowlist, no loops, 40s timeout per grading job in a separate process, 200 KB request cap.
