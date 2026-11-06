"""Admin CLI. Works directly on the database; the server can be running.

python manage.py add "Team Name"          add one team, print its token
python manage.py import teams.txt         one team name per line; writes tokens.csv
python manage.py list                     teams, tokens, members (use this when someone loses a token)
python manage.py reveal on|off            show/hide A4 scores on the leaderboard
python manage.py freeze on|off            stop/allow submissions
"""
import csv
import sys

import server


def main(argv):
    server.init_db()
    if len(argv) < 2:
        print(__doc__); return
    cmd = argv[1]
    if cmd == "add":
        print(server.add_team(argv[2]))
    elif cmd == "import":
        names = [l.strip() for l in open(argv[2]) if l.strip()]
        with open("tokens.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["team", "token"])
            for n in names:
                w.writerow([n, server.add_team(n)])
        print(f"Added {len(names)} teams. Tokens in tokens.csv")
    elif cmd == "list":
        with server.db() as con:
            for r in con.execute("SELECT name, token, members FROM teams ORDER BY name"):
                print(f"{r['name']:<30} {r['token']:<14} {r['members'] or ''}")
    elif cmd in ("reveal", "freeze"):
        server.set_setting("reveal" if cmd == "reveal" else "frozen", "1" if argv[2] == "on" else "0")
        print(f"{cmd} {argv[2]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
