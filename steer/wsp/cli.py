"""The programmer's CLI for the weight-space programming lab.

  wsp status                         budget, rules, architecture, candidate dials
  wsp kernel  < code.py              run Python in the persistent weight kernel (or: wsp kernel -c "print(1)")
  wsp forward < request.json         spend one forward pass (generate or read)
"""

import json
import os
import sys

import httpx

URL = os.environ.get("WSP_LAB", "http://127.0.0.1:8790")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "kernel", "forward"):
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    c = httpx.Client(base_url=URL, timeout=3600)
    if cmd == "status":
        print(json.dumps(c.get("/status").json(), indent=2))
    elif cmd == "kernel":
        code = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "-c" else sys.stdin.read()
        r = c.post("/kernel", json={"code": code}).json()
        print(r["output"], end="" if r["output"].endswith("\n") else "\n")
        if not r["ok"]:
            sys.exit(1)
    elif cmd == "forward":
        req = json.loads(sys.stdin.read())
        r = c.post("/forward", json=req)
        print(json.dumps(r.json(), indent=1, ensure_ascii=False))
        if r.status_code != 200:
            sys.exit(1)


if __name__ == "__main__":
    main()
