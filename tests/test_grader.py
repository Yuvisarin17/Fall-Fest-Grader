"""End-to-end check. With a server running and two team tokens:

    FALLFEST_SERVER=http://localhost:8000 python tests/test_grader.py TOKEN_A TOKEN_B

Every reference solution must pass (or score), every wrong answer must fail,
and the abuse cases must be rejected cleanly."""
import contextlib
import io
import os
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "client"), str(ROOT / "tests")]

import fallfest as ff  # noqa: E402
import refs  # noqa: E402
import refs_adv as RA  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

failures = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        failures.append(label)


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a)


def expect_error(fn, *a):
    try:
        quiet(fn, *a)
        return None
    except Exception as e:
        return str(e)


def main(tok_a, tok_b):
    quiet(ff.login, tok_a)
    print("Beginner references")
    for cid, f in refs.PASS.items():
        for fn in [f] + refs.PASS_EXTRA.get(cid, []):
            r = quiet(ff.submit, cid, fn())
            check(r["passed"], f"{cid} {fn.__name__} passes")
        for w in refs.WRONG.get(cid, []):
            try:
                r = quiet(ff.submit, cid, w())
                check(not r["passed"], f"{cid} wrong answer fails: {r['message'][:70]}")
            except RuntimeError as e:
                check(True, f"{cid} wrong answer rejected: {str(e)[:70]}")

    print("Advanced references")
    for cid, sub in [("a1", RA.a1()), ("a2", RA.a2()), ("a3", RA.a3()), ("a5", RA.a5(6)),
                     ("a6", RA.a6_coherent()), ("a6", RA.a6_dynamic()), ("a7", RA.a7())]:
        r = quiet(ff.submit, cid, sub)
        check(r["valid"] and r["score"] is not None, f"{cid} scores {r['score']:.4g}")
    r = quiet(ff.submit, "a6", RA.a6_cheat_classical_copy())
    check(not r["valid"], "a6 classical-copy cheat does not count")

    print("A4 noisy backend")
    tgt = quiet(ff.a4_target)
    tgt.measure_all()
    counts = quiet(ff.run_noisy, [tgt], 20000)[0]
    est = sum((-1) ** k.replace(" ", "").count("1") * v for k, v in counts.items()) / 20000
    r = quiet(ff.submit, "a4", est)
    check(r["score"] is None and r.get("hidden"), "a4 score hidden from participant")
    err = expect_error(ff.run_noisy, [tgt], 100_000)
    err = err or expect_error(ff.run_noisy, [tgt] * 20, 100_000)
    check(err is not None and "left" in err, "a4 shot budget enforced")

    print("Abuse and edge cases")
    check("qubits" in (expect_error(ff.submit, "b1", QuantumCircuit(30)) or ""), "qubit cap")
    big = QuantumCircuit(1)
    for _ in range(8000):
        big.x(0)
    check(expect_error(ff.submit, "b1", big) is not None, "oversized circuit rejected")
    check("OpenQASM" in (expect_error(ff.submit, "b1", "import os; os.system('rm -rf /')") or ""), "python text rejected")
    nest = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + "gate g0 a { x a; x a; }\n" + "".join(
        f"gate g{i} a {{ g{i-1} a; g{i-1} a; g{i-1} a; g{i-1} a; }}\n" for i in range(1, 12)) + "qreg q[1];\ng11 q[0];\n"
    t = time.time()
    check(expect_error(ff.submit, "b1", nest) is not None and time.time() - t < 30, "exponential gate nesting rejected")
    check(expect_error(ff.submit, "b99", refs.b1()) is not None, "unknown challenge")
    check(expect_error(ff.submit, "a2", {"gammas": [1e9], "betas": [0]}) is not None, "non-finite params rejected")
    p = QuantumCircuit(1); p.x(0); p.measure_all()
    from qiskit.circuit import Parameter
    pc = QuantumCircuit(1); pc.rx(Parameter("t"), 0)
    check("unbound" in (expect_error(ff.submit, "b4", pc) or ""), "unbound parameters caught client-side")
    old = ff._TOKEN; ff._TOKEN = "nope"
    check("Unknown team" in (expect_error(ff.submit, "b1", refs.b1()) or ""), "bad token rejected")
    ff._TOKEN = old

    print("Second team, leaderboard")
    quiet(ff.login, tok_b)
    quiet(ff.submit, "b1", refs.b1())
    quiet(ff.submit, "a2", {"gammas": [0.3], "betas": [0.3]})
    import requests
    lb = requests.get(ff.SERVER + "/api/leaderboard").json()
    a2 = lb["advanced"]["a2"]["entries"]
    check(len(a2) == 2 and a2[0]["score"] >= a2[1]["score"], "a2 ranked high-to-low")
    check(lb["advanced"]["a4"]["hidden"] and all(e["score"] is None for e in lb["advanced"]["a4"]["entries"]),
          "a4 hidden on leaderboard")
    done = {t["team"]: len(t["done"]) for t in lb["beginner"]}
    check(max(done.values()) == 14, f"beginner progress recorded {done}")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
