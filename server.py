"""Qiskit Fall Fest 2026 grader server.

Run:  ADMIN_TOKEN=... uvicorn server:app --host 0.0.0.0 --port 8000
Teams: python manage.py add "Team Name"   (prints the team token)
"""
from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pebble import ProcessPool
from pydantic import BaseModel

from grader import registry
from grader.assets import A4_SHOT_BUDGET

ROOT = Path(__file__).resolve().parent
DB_PATH = os.environ.get("FALLFEST_DB", str(ROOT / "fallfest.db"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
JOIN_CODE = os.environ.get("JOIN_CODE", "")
WORKERS = int(os.environ.get("GRADER_WORKERS", "2"))
TIMEOUT_S = float(os.environ.get("GRADER_TIMEOUT", "40"))
MIN_GAP_S = float(os.environ.get("SUBMIT_MIN_GAP", "3"))
MAX_BODY = 200_000
MAX_RUN_CIRCUITS = 20
MAX_SHOTS_PER_CALL = 100_000

app = FastAPI(title="Qiskit Fall Fest Grader", docs_url=None, redoc_url=None)
_db_lock = threading.Lock()
_last_submit: dict[int, float] = {}
pool: ProcessPool | None = None


# ---------- database ----------

def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS teams(
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, token TEXT UNIQUE NOT NULL,
            seed INTEGER NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS submissions(
            id INTEGER PRIMARY KEY, team_id INTEGER NOT NULL, challenge TEXT NOT NULL, ts REAL NOT NULL,
            passed INTEGER, score REAL, valid INTEGER, tiebreak REAL, badge TEXT, message TEXT, extra TEXT);
        CREATE INDEX IF NOT EXISTS sub_tc ON submissions(team_id, challenge);
        CREATE TABLE IF NOT EXISTS shots(team_id INTEGER PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        """)
        cols = {r["name"] for r in con.execute("PRAGMA table_info(teams)")}
        if "members" not in cols:
            con.execute("ALTER TABLE teams ADD COLUMN members TEXT")
        scols = {r["name"] for r in con.execute("PRAGMA table_info(submissions)")}
        if "payload" not in scols:
            con.execute("ALTER TABLE submissions ADD COLUMN payload TEXT")


def setting(key, default=None):
    with db() as con:
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key, value):
    with _db_lock, db() as con:
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value))


def add_team(name: str, members: str = "") -> str:
    token = secrets.token_urlsafe(9)
    seed = secrets.randbelow(2 ** 31)
    with _db_lock, db() as con:
        con.execute("INSERT INTO teams(name, token, seed, created, members) VALUES(?,?,?,?,?)",
                    (name, token, seed, time.time(), members))
    return token


def team_for(auth: str | None) -> sqlite3.Row:
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing team token. Call login('YOUR-TEAM-TOKEN').")
    with db() as con:
        t = con.execute("SELECT * FROM teams WHERE token=?", (auth[7:].strip(),)).fetchone()
    if not t:
        raise HTTPException(401, "Unknown team token. Check with a mentor.")
    return t


def require_admin(auth: str | None):
    if not ADMIN_TOKEN or auth != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(403, "Admin only.")


# ---------- lifecycle ----------

@app.on_event("startup")
def startup():
    global pool
    init_db()
    pool = ProcessPool(max_workers=WORKERS, max_tasks=200, initializer=registry.warm_up)


@app.on_event("shutdown")
def shutdown():
    if pool:
        pool.close()


@app.middleware("http")
async def limit_body(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_BODY:
        return JSONResponse({"detail": "Submission too large."}, status_code=413)
    return await call_next(request)


def run_in_pool(fn, *args):
    fut = pool.schedule(fn, args=args, timeout=TIMEOUT_S)
    try:
        return fut.result()
    except FutureTimeout:
        raise HTTPException(408, f"Grading took longer than {TIMEOUT_S:.0f}s. Make your circuit smaller.")
    except Exception:
        raise HTTPException(500, "Grader worker crashed on this submission. Tell a mentor.")


def throttle(team_id: int):
    now = time.time()
    last = _last_submit.get(team_id, 0)
    if now - last < MIN_GAP_S:
        wait = max(1, math.ceil(MIN_GAP_S - (now - last)))
        raise HTTPException(429, f"Slow down: wait {wait}s between submissions.", headers={"Retry-After": str(wait)})
    _last_submit[team_id] = now


# ---------- participant API ----------

class Submission(BaseModel):
    challenge: str
    data: object


class RunRequest(BaseModel):
    challenge: str = "a4"
    circuits: list[str]
    shots: int = 4000


@app.post("/api/submit")
def submit(sub: Submission, authorization: str | None = Header(None)):
    team = team_for(authorization)
    cid = sub.challenge.lower().strip()
    if cid not in registry.META:
        raise HTTPException(404, f"No challenge called '{sub.challenge}'.")
    if setting("frozen") == "1":
        raise HTTPException(403, "Submissions are closed.")
    throttle(team["id"])
    res = run_in_pool(registry.grade, cid, sub.data, {"team_seed": team["seed"]})
    if "error" in res:
        if "internal" in res:
            print(f"[internal error] team={team['name']} {cid}\n{res['internal']}", flush=True)
        raise HTTPException(400, res["error"])
    score = res.get("score")
    if score is not None and not math.isfinite(score):
        score = None
    extra = {k: v for k, v in res.items() if k not in ("passed", "score", "valid", "message", "badge", "tiebreak")}
    with _db_lock, db() as con:
        con.execute("""INSERT INTO submissions(team_id, challenge, ts, passed, score, valid, tiebreak, badge, message, extra, payload)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (team["id"], cid, time.time(), int(res["passed"]) if "passed" in res else None, score,
                     int(res.get("valid", True)), res.get("tiebreak"), res.get("badge"), res["message"],
                     json.dumps(extra, default=str), json.dumps(sub.data, default=str)[:100_000]))
    out = {k: v for k, v in res.items() if k not in ("internal",)}
    if res.get("hidden"):
        out["score"] = None  # never leak a hidden score
    elif score is None and "score" in res:
        out["score"] = None
    return out


@app.post("/api/run")
def run(req: RunRequest, authorization: str | None = Header(None)):
    team = team_for(authorization)
    if req.challenge.lower() != "a4":
        raise HTTPException(400, "The noisy simulator is only for A4.")
    if not req.circuits or len(req.circuits) > MAX_RUN_CIRCUITS:
        raise HTTPException(400, f"Send between 1 and {MAX_RUN_CIRCUITS} circuits per call.")
    if not 1 <= req.shots <= MAX_SHOTS_PER_CALL:
        raise HTTPException(400, f"Shots per circuit must be between 1 and {MAX_SHOTS_PER_CALL:,}.")
    cost = req.shots * len(req.circuits)
    with _db_lock, db() as con:
        con.execute("INSERT OR IGNORE INTO shots(team_id, used) VALUES(?,0)", (team["id"],))
        used = con.execute("SELECT used FROM shots WHERE team_id=?", (team["id"],)).fetchone()["used"]
        if used + cost > A4_SHOT_BUDGET:
            raise HTTPException(429, f"That needs {cost:,} shots but you have {A4_SHOT_BUDGET - used:,} left.")
        con.execute("UPDATE shots SET used=used+? WHERE team_id=?", (cost, team["id"]))
    res = run_in_pool(registry.run_noisy, req.circuits, req.shots, {"run_seed": secrets.randbelow(2 ** 31)})
    if "error" in res:
        with _db_lock, db() as con:  # refund
            con.execute("UPDATE shots SET used=used-? WHERE team_id=?", (cost, team["id"]))
        raise HTTPException(400, res["error"])
    return {"counts": res["counts"], "shots_left": A4_SHOT_BUDGET - used - cost}


@app.get("/api/challenges")
def challenges():
    return [{"id": k, "title": v[0], "kind": registry.kind(k)} for k, v in registry.META.items()]


@app.get("/api/challenge/{cid}")
def challenge(cid: str, authorization: str | None = Header(None)):
    team = team_for(authorization)
    cid = cid.lower()
    if cid not in registry.META:
        raise HTTPException(404, "No such challenge.")
    return registry.info(cid, team["seed"])


@app.get("/api/me")
def me(authorization: str | None = Header(None)):
    team = team_for(authorization)
    with db() as con:
        passed = [r["challenge"] for r in con.execute(
            "SELECT DISTINCT challenge FROM submissions WHERE team_id=? AND passed=1", (team["id"],))]
        used = con.execute("SELECT used FROM shots WHERE team_id=?", (team["id"],)).fetchone()
    return {"team": team["name"], "passed": sorted(passed, key=lambda c: int(c[1:])),
            "a4_shots_left": A4_SHOT_BUDGET - (used["used"] if used else 0)}


# ---------- leaderboard ----------

def leaderboard_data(reveal_hidden: bool) -> dict:
    with db() as con:
        teams = {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM teams")}
        rows = con.execute("SELECT * FROM submissions ORDER BY ts").fetchall()
    beg = {name: {"done": [], "badges": []} for name in teams.values()}
    best: dict[str, dict[str, dict]] = {}
    last_a4: dict[str, dict] = {}
    for r in rows:
        name = teams.get(r["team_id"])
        if name is None:
            continue
        cid = r["challenge"]
        if registry.kind(cid) == "pass":
            if r["passed"] and cid not in beg[name]["done"]:
                beg[name]["done"].append(cid)
            if r["badge"] and r["badge"] not in beg[name]["badges"]:
                beg[name]["badges"].append(r["badge"])
            continue
        if cid == "a4":
            last_a4[name] = {"score": r["score"], "ts": r["ts"]}
            continue
        if not r["valid"] or r["score"] is None:
            continue
        better = registry.META[cid][1]
        cur = best.setdefault(cid, {}).get(name)
        key = (r["score"] if better == "low" else -r["score"], r["tiebreak"] or 0)
        if cur is None or key < cur["key"]:
            best[cid][name] = {"score": r["score"], "tiebreak": r["tiebreak"], "ts": r["ts"], "key": key}
    adv = {}
    for cid in [c for c in registry.META if registry.kind(c) == "score"]:
        if cid == "a4":
            entries = [{"team": n, "score": v["score"] if reveal_hidden else None} for n, v in last_a4.items()]
            if reveal_hidden:
                entries.sort(key=lambda e: e["score"])
            else:
                entries.sort(key=lambda e: e["team"].lower())
        else:
            entries = sorted(({"team": n, "score": v["score"], "tiebreak": v["tiebreak"], "key": v["key"], "ts": v["ts"]}
                              for n, v in best.get(cid, {}).items()), key=lambda e: (e["key"], e["ts"]))
            for e in entries:
                e.pop("key"); e.pop("ts")
        adv[cid] = {"title": registry.META[cid][0], "better": registry.META[cid][1],
                    "hidden": cid == "a4" and not reveal_hidden, "entries": entries}
    beginners = [{"team": n, "done": sorted(v["done"], key=lambda c: int(c[1:])), "badges": v["badges"]}
                 for n, v in sorted(beg.items(), key=lambda kv: kv[0].lower())]
    return {"beginner": beginners, "advanced": adv, "updated": time.time()}


@app.get("/api/leaderboard")
def leaderboard():
    return leaderboard_data(setting("reveal") == "1")


BEGINNER_NB = os.environ.get("BEGINNER_NOTEBOOK", "")
ADVANCED_NB = os.environ.get("ADVANCED_NOTEBOOK", "")


@app.get("/", response_class=HTMLResponse)
def hub_page():
    import html as _html
    btns = []
    for label, url in (("Beginner notebook", BEGINNER_NB), ("Advanced notebook", ADVANCED_NB)):
        if url:
            btns.append(f'<a class="btn" href="{_html.escape(url, quote=True)}" target="_blank" rel="noopener">{label}</a>')
    nb = " ".join(btns) if btns else '<p class="soon">Notebook links are on the screen at the front.</p>'
    return (ROOT / "static" / "hub.html").read_text(encoding="utf-8").replace("{{NOTEBOOK_BUTTONS}}", nb)


@app.get("/leaderboard", response_class=HTMLResponse)
def board_page():
    return (ROOT / "static" / "leaderboard.html").read_text(encoding="utf-8")


@app.get("/fallfest.py", response_class=PlainTextResponse)
def client_file():
    return (ROOT / "client" / "fallfest.py").read_text(encoding="utf-8")


# ---------- self-serve sign-up ----------

_join_hits: dict[str, list[float]] = {}


class JoinIn(BaseModel):
    name: str
    members: str = ""
    code: str


@app.get("/join", response_class=HTMLResponse)
def join_page():
    return (ROOT / "static" / "join.html").read_text(encoding="utf-8")


@app.post("/api/join")
def join(j: JoinIn, request: Request):
    if not JOIN_CODE:
        raise HTTPException(403, "Sign-up is closed. Ask an organizer.")
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = request.headers.get("cf-connecting-ip") or fwd or (request.client.host if request.client else "?")
    now = time.time()
    hits = [t for t in _join_hits.get(ip, []) if now - t < 600]
    if len(hits) >= 8:
        raise HTTPException(429, "Too many sign-ups from this device. Ask an organizer.")
    _join_hits[ip] = hits + [now]
    if not secrets.compare_digest(j.code.strip().lower(), JOIN_CODE.strip().lower()):
        raise HTTPException(403, "Wrong event code. It's on the screen at the front.")
    name = " ".join(j.name.split())
    if not 2 <= len(name) <= 40:
        raise HTTPException(400, "Name must be 2 to 40 characters.")
    members = " ".join(j.members.split())[:300]
    try:
        token = add_team(name, members)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"'{name}' is already taken. If that's you and you lost your token, ask an organizer.")
    return {"name": name, "token": token}


@app.get("/healthz")
def health():
    return {"ok": True}


# ---------- admin ----------

class TeamIn(BaseModel):
    name: str


@app.post("/admin/teams")
def admin_add_team(t: TeamIn, authorization: str | None = Header(None)):
    require_admin(authorization)
    try:
        return {"name": t.name, "token": add_team(t.name)}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Team name already exists.")


@app.post("/admin/setting/{key}/{value}")
def admin_setting(key: str, value: str, authorization: str | None = Header(None)):
    require_admin(authorization)
    if key not in ("reveal", "frozen"):
        raise HTTPException(400, "Settings: reveal, frozen.")
    set_setting(key, value)
    return {key: value}


# ---------- integrity report ----------

# Challenges where independent solutions rarely come out identical.
VARIED = {"b12", "b13", "b14", "a1", "a3", "a5", "a6", "a7"}
HARD = {"b10", "b11", "b12", "b13", "b14", "a1", "a3", "a5", "a6", "a7"}
STREAK_WINDOW_S = 600
STREAK_COUNT = 4


def _norm(payload: str | None) -> str | None:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if isinstance(data, list):
        data = "\n".join(map(str, data))
    if not isinstance(data, str):
        return None
    lines = [" ".join(l.split()) for l in data.splitlines()]
    return "\n".join(l for l in lines if l and not l.startswith(("OPENQASM", "include")))


def integrity_report() -> list[dict]:
    with db() as con:
        teams = {r["id"]: dict(r) for r in con.execute("SELECT id, name, members, created FROM teams")}
        subs = [dict(r) for r in con.execute(
            "SELECT id, team_id, challenge, ts, passed, valid, payload FROM submissions ORDER BY ts")]
    first_seen: dict[tuple, tuple] = {}
    out = {tid: {"team": t["name"], "members": t["members"] or "", "joined": t["created"], "flags": [],
                 "passes": [], "attempts": {}} for tid, t in teams.items()}
    for s in subs:
        r = out.get(s["team_id"])
        if r is None:
            continue
        cid = s["challenge"]
        a = r["attempts"].setdefault(cid, {"fails": 0, "first_ok": None})
        ok = bool(s["passed"]) if s["passed"] is not None else bool(s["valid"])
        if ok and a["first_ok"] is None:
            a["first_ok"] = s["ts"]
            a["fails_before"] = a["fails"]
            if s["passed"] is not None:
                r["passes"].append((s["ts"], cid))
        elif not ok:
            a["fails"] += 1
        if cid in VARIED and ok:
            key = (cid, _norm(s["payload"]))
            if key[1]:
                prev = first_seen.get(key)
                if prev is None:
                    first_seen[key] = (s["team_id"], s["ts"])
                elif prev[0] != s["team_id"]:
                    other = teams[prev[0]]["name"]
                    msg = f"{cid.upper()}: identical circuit to {other} (submitted {int((s['ts'] - prev[1]) / 60)} min after them)"
                    if msg not in r["flags"]:
                        r["flags"].append(msg)
    for r in out.values():
        hard_ts = sorted(ts for ts, cid in r["passes"] if cid in HARD or int(cid[1:]) >= 7)
        for i in range(len(hard_ts) - STREAK_COUNT + 1):
            if hard_ts[i + STREAK_COUNT - 1] - hard_ts[i] <= STREAK_WINDOW_S:
                r["flags"].append(f"Fast streak: {STREAK_COUNT} Part B challenges passed within "
                                  f"{int((hard_ts[i + STREAK_COUNT - 1] - hard_ts[i]) / 60)} min")
                break
        clean = sorted(c for c, a in r["attempts"].items()
                       if c in HARD and a["first_ok"] is not None and a.get("fails_before", 0) == 0)
        if len(clean) >= 3:
            r["flags"].append(f"First try, no failed attempts on {len(clean)} hard challenges: {', '.join(c.upper() for c in clean)}")
        done = {c: a["first_ok"] for c, a in r["attempts"].items() if a["first_ok"] is not None}
        beg = [f"b{i}" for i in range(1, 15)]
        adv = [c for c in registry.META if registry.kind(c) == "score"]
        first = min(done.values()) if done else None
        r["finish_beginner"] = max(done[c] for c in beg) if all(c in done for c in beg) else None
        r["finish_all"] = max(done[c] for c in beg + adv) if all(c in done for c in beg + adv) else None
        r["first_submit"] = first
        r["beg_done"] = sum(c in done for c in beg)
        r["adv_done"] = sum(c in done for c in adv)
        r["solved"] = len(done)
        r["fails"] = sum(a["fails"] for a in r["attempts"].values())
        r["timeline"] = [(int((ts - r["joined"]) / 60), cid.upper()) for ts, cid in r["passes"]]
        del r["attempts"], r["passes"]
    return sorted(out.values(), key=lambda r: (-len(r["flags"]), r["team"].lower()))


@app.get("/admin/report", response_class=HTMLResponse)
def admin_report(key: str = ""):
    if not ADMIN_TOKEN or not secrets.compare_digest(key, ADMIN_TOKEN):
        raise HTTPException(404, "Not Found")
    import html as _h
    report = integrity_report()

    def speed_table(field, label, progress):
        fin = sorted((r for r in report if r[field]), key=lambda r: r[field])
        body = ""
        for i, r in enumerate(fin, 1):
            took = int((r[field] - r["first_submit"]) / 60)
            fl = "; ".join(_h.escape(f) for f in r["flags"]) or '<span class="ok">none</span>'
            body += (f'<tr class="{"hot" if r["flags"] else ""}"><td class="rk">{i}</td><td><b>{_h.escape(r["team"])}</b>'
                     f'<div class="m">{_h.escape(r["members"])}</div></td>'
                     f'<td><span class="t" data-ts="{r[field]}"></span></td><td>{took} min</td><td class="fl">{fl}</td></tr>')
        if not body:
            close = sorted(report, key=lambda r: -progress(r))[:5]
            near = ", ".join(f"{_h.escape(r['team'])} ({progress(r)})" for r in close if progress(r)) or "nobody yet"
            body = f'<tr><td colspan="5" class="m">No finishers yet. Closest: {near}</td></tr>'
        return (f"<h2>{label}</h2><table><tr><th>#</th><th>Team</th><th>Finished at</th>"
                f"<th>First to last submission</th><th>Integrity flags</th></tr>{body}</table>")

    speed = (speed_table("finish_beginner", "Speed track: all 14 beginner challenges", lambda r: r["beg_done"])
             + speed_table("finish_all", "Speed track: everything (14 beginner + 7 advanced)",
                           lambda r: r["beg_done"] + r["adv_done"]))
    rows = []
    for r in report:
        flags = "".join(f"<li>{_h.escape(f)}</li>" for f in r["flags"]) or '<span class="ok">none</span>'
        tl = " · ".join(f"{_h.escape(c)} @{m}m" for m, c in r["timeline"]) or "-"
        rows.append(f'<tr class="{"hot" if r["flags"] else ""}"><td><b>{_h.escape(r["team"])}</b>'
                    f'<div class="m">{_h.escape(r["members"])}</div></td><td>{r["solved"]}</td><td>{r["fails"]}</td>'
                    f'<td><ul>{flags}</ul></td><td class="m">{tl}</td></tr>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Integrity report</title><style>
body{{background:#0d0f1a;color:#eef0f8;font-family:system-ui,sans-serif;margin:0;padding:24px}}
h1{{margin:0 0 4px}} p{{color:#8a90ab;max-width:820px}} table{{border-collapse:collapse;width:100%;margin-top:16px}}
td,th{{text-align:left;vertical-align:top;padding:10px 8px;border-bottom:1px solid #262b44;font-size:14px}}
th{{color:#8a90ab;font-weight:600}} tr.hot td{{background:rgba(255,131,137,.07)}} ul{{margin:0;padding-left:18px;color:#ffb3b8}}
.m{{color:#8a90ab;font-size:13px}} .ok{{color:#42be65}} h2{{margin:28px 0 0;font-size:18px}}
td.rk{{font-weight:700;color:#f1c21b;width:28px}} td.fl{{color:#ffb3b8;font-size:13px}}
</style></head><body><h1>Integrity report</h1>
<p>Speed track ranks by the clock time of each team's last required completion (advanced counts once a valid
submission is in). Flags mean "worth a conversation", not "cheated". Fast learners and experienced students trip the speed flags too.
Timeline shows minutes after sign-up for each beginner pass. Ask flagged teams to explain one solution out loud.
Refreshes every minute.</p>
{speed}
<h2>All teams</h2>
<table><tr><th>Team</th><th>Solved</th><th>Failed tries</th><th>Flags</th><th>Timeline</th></tr>{"".join(rows)}</table>
<script>document.querySelectorAll(".t").forEach(e=>{{e.textContent=new Date(+e.dataset.ts*1000).toLocaleTimeString([], {{hour:"numeric",minute:"2-digit",second:"2-digit"}})}})</script>
</body></html>"""


@app.get("/admin/teams")
def admin_teams(authorization: str | None = Header(None)):
    require_admin(authorization)
    with db() as con:
        return [dict(r) for r in con.execute("SELECT name, token, members FROM teams ORDER BY name")]


@app.get("/admin/submissions")
def admin_submissions(authorization: str | None = Header(None), team: str | None = None, limit: int = 200):
    require_admin(authorization)
    q = "SELECT s.*, t.name AS team FROM submissions s JOIN teams t ON t.id=s.team_id"
    args = []
    if team:
        q += " WHERE t.name=?"; args.append(team)
    q += " ORDER BY s.ts DESC LIMIT ?"; args.append(limit)
    with db() as con:
        return [dict(r) for r in con.execute(q, args)]