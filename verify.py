"""Re-grade every code teams pasted into the Google Form and build the leaderboard.

1. Google Form with two questions: "Team name" (short answer) and "Submission code" (paragraph).
2. Responses > Download CSV (or link a Sheet and File > Download > CSV).
3. python verify.py responses.csv            -> leaderboard.html + results.csv
   python verify.py responses.csv --reveal   -> also ranks A4 (do this at the end)

Nothing is trusted from the participant's notebook: each code is decoded and graded again here.
"""
import base64
import csv
import html
import json
import sys
import zlib

from grader import registry

A4_SEED = 2026


def decode(code: str):
    code = code.strip()
    if not code.startswith("FF1:"):
        raise ValueError("not a Fall Fest code")
    raw = zlib.decompress(base64.urlsafe_b64decode(code[4:].encode()), bufsize=1 << 20)
    if len(raw) > 300_000:
        raise ValueError("code too large")
    obj = json.loads(raw)
    return obj["c"], obj["d"]


def find_col(header, *words):
    for i, h in enumerate(header):
        if any(w in h.lower() for w in words):
            return i
    raise SystemExit(f"Couldn't find a column containing {words} in {header}")


def main(path, reveal):
    rows = list(csv.reader(open(path, newline="", encoding="utf-8-sig")))
    header, rows = rows[0], rows[1:]
    ti, ci = find_col(header, "team"), find_col(header, "code", "submission")
    done, badges, best, a4, log = {}, {}, {}, {}, []
    for n, r in enumerate(rows, start=2):
        team = r[ti].strip()
        if not team:
            continue
        try:
            cid, data = decode(r[ci])
            if cid not in registry.META:
                raise ValueError(f"unknown challenge {cid}")
            res = registry.grade(cid, data, {"team_seed": A4_SEED})
        except Exception as e:
            log.append((n, team, "?", f"bad code: {e}"))
            continue
        if "error" in res:
            log.append((n, team, cid, "rejected: " + res["error"]))
            continue
        done.setdefault(team, set())
        if "passed" in res:
            if res["passed"]:
                done[team].add(cid)
                if res.get("badge"):
                    badges.setdefault(team, set()).add(res["badge"])
            log.append((n, team, cid, "pass" if res["passed"] else "FAIL"))
            continue
        if cid == "a4":
            a4[team] = res["score"]  # last submission counts
            log.append((n, team, cid, "a4 recorded"))
            continue
        if not res.get("valid", True):
            log.append((n, team, cid, "does not count: " + res["message"]))
            continue
        low = registry.META[cid][1] == "low"
        key = (res["score"] if low else -res["score"], res.get("tiebreak") or 0, n)
        cur = best.setdefault(cid, {}).get(team)
        if cur is None or key < cur[0]:
            best[cid][team] = (key, res["score"])
        log.append((n, team, cid, f"score {res['score']:.6g}"))

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["form_row", "team", "challenge", "result"])
        w.writerows(log)

    adv = {}
    for cid in (c for c in registry.META if registry.kind(c) == "score"):
        if cid == "a4":
            items = sorted(a4.items(), key=lambda kv: kv[1]) if reveal else sorted((t, None) for t in a4)
        else:
            items = [(t, v[1]) for t, v in sorted(best.get(cid, {}).items(), key=lambda kv: kv[1][0])]
        adv[cid] = items
    write_html(done, badges, adv, reveal)
    print(f"{len(rows)} form rows -> leaderboard.html, results.csv")
    for cid, items in adv.items():
        if items:
            print(f"  {cid.upper():4} " + ", ".join(f"{t} ({'?' if s is None else f'{s:.4g}'})" for t, s in items[:3]))


def write_html(done, badges, adv, reveal):
    e = html.escape
    cols = [f"b{i}" for i in range(1, 15)]
    rows = ""
    for t in sorted(done, key=str.lower):
        b = "".join(f'<span class="badge">{e(x)}</span>' for x in sorted(badges.get(t, ())))
        cells = "".join(f'<td><span class="dot{" on" if c in done[t] else ""}{" b" if i >= 6 else ""}"></span></td>'
                        for i, c in enumerate(cols))
        rows += f'<tr><td class="team">{e(t)}{b}</td>{cells}<td class="count">{len(done[t] & set(cols))}/14</td></tr>'
    cards = ""
    for cid, items in adv.items():
        title, better = registry.META[cid]
        hidden = cid == "a4" and not reveal
        sub = "scores revealed at the end" if hidden else ("lower is better" if better == "low" else "higher is better")
        lis = "".join(
            f'<li><span class="rank {"" if hidden else f"r{i+1}"}">{"·" if hidden else i+1}</span>'
            f'<span class="name">{e(t)}</span><span class="score">{"submitted" if s is None else (int(s) if float(s).is_integer() else f"{s:.4g}")}</span></li>'
            for i, (t, s) in enumerate(items[:10])) or '<div class="empty">No valid submissions yet</div>'
        cards += f'<div class="panel card"><h3>{cid.upper()} · {e(title)}</h3><div class="sub">{sub}</div><ol>{lis}</ol></div>'
    head = "".join(f"<th>{c[1:]}</th>" for c in cols)
    tpl = open("static/board_template.html").read()
    out = tpl.replace("{{HEAD}}", head).replace("{{ROWS}}", rows or '<tr><td class="team">No submissions yet</td></tr>')
    open("leaderboard.html", "w").write(out.replace("{{CARDS}}", cards))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], "--reveal" in sys.argv)
