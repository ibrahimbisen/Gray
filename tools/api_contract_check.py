"""Every field a page reads must exist in the payload the server sends it.

    python tools/api_contract_check.py

Exits non-zero if any page reads a key the server never sends.

This is the failure mode the dashboard is most prone to, because it is silent:
the data shape changes, a page still reads the old key, and the panel renders
blank with nothing in any console to say why. Nobody notices until they go
looking for a number that used to be there.

It works by scraping `<var>.<field>` out of each page's script and checking it
against the real dict the corresponding server function returns - so it runs the
actual code rather than trusting a schema written alongside it.

**This tool was broken for the whole of the last rebuild.** It read index.html,
monitor.html, stage1/2/3.html, which moved into dashboard/archive/, so it raised
FileNotFoundError on its first line of work and nobody saw the message. A checker
that cannot run is worse than no checker, because the repo still believes it is
covered. Missing pages are now REPORTED, not raised on, so the same thing cannot
happen quietly a second time.

Note the deliberate consequence: shadowing a page's state variable inside that
page makes this report false positives. That is a feature. The name means the
page state in every one of these files, and a second meaning for it is worth
renaming.

Pair with tools/render_check.js, which executes the panels instead of reading
them - between them, a broken page shows up here rather than in the morning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import server  # noqa: E402

HERE = ROOT / "dashboard"

# Every payload, built once. Named, because six pages share four of them and
# building monitor_state() three times would only slow the check down.
PAYLOADS = {
    "/api/state": server.summary_state,
    "/api/monitor": server.monitor_state,
    "/api/overview": server.overview_state,
    "/api/live": server.live_state,
    "/api/week": server.week_state,
    "/api/carry": server.carry_state,
    "/api/dials": server.dials_state,
    "/api/controls": server.controls_state,
    "/api/stage/1": lambda: server.stage_state(1),
    "/api/stage/2": lambda: server.stage_state(2),
    "/api/stage/3": lambda: server.stage_state(3),
}

# page -> {the variable the page holds the payload in: the endpoint it came from}
#
# Three pages hold more than one payload at a time, which is why this is a map
# per page rather than one `s` for everything. control.html reads S, M and L in
# the same expression, and getting them the wrong way round is exactly the kind
# of fault this file exists to catch.
PAGES = {
    "now.html": {"S": "/api/overview", "M": "/api/monitor", "L": "/api/live"},
    "runs.html": {"STATE": "/api/monitor"},
    "plan.html": {"s": "/api/week"},
    "train.html": {"s": "/api/stage/2"},
    "dials.html": {"s": "/api/dials"},
    "robot.html": {"S1": "/api/stage/1", "S3": "/api/stage/3"},
    "controller.html": {"s": "/api/controls"},
    "summary.html": {"s": "/api/state"},
    "carry.html": {"s": "/api/carry"},
}


def main() -> int:
    cache: dict[str, dict] = {}

    def payload(name: str) -> dict:
        if name not in cache:
            cache[name] = PAYLOADS[name]()
        return cache[name]

    broken, absent, dead_rows = 0, 0, []
    for page, bindings in PAGES.items():
        path = HERE / page
        if not path.is_file():
            absent += 1
            print(f"{page:<17} NOT FOUND - remove it from PAGES or put the file back")
            continue
        text = path.read_text(encoding="utf-8")

        for var, endpoint in bindings.items():
            got = payload(endpoint)
            # (?<![\w.]) and not \b: `tr.dataset.s.includes(term)` is not a read
            # of the payload, but \b matches after the dot and reported it as
            # one. The lookbehind says "s must not be part of a longer name and
            # must not itself be a property of something else".
            used = sorted(set(re.findall(
                rf"(?<![\w.]){re.escape(var)}\.([A-Za-z_]\w*)", text)))
            missing = [f for f in used if f not in got]
            label = f"{page} {var}"
            if missing:
                broken += 1
            print(f"{label:<26} {endpoint:<15} reads {len(used):>2} fields -> "
                  f"{'MISSING ' + ', '.join(missing) if missing else 'ok'}")
            if missing:
                print(f"{'':<26} all of them: {', '.join(used)}")
            dead = sorted(set(got) - set(used))
            if dead:
                dead_rows.append(f"  {label:<26} {', '.join(dead)}")

    if dead_rows:
        print("\nkeys the server sends that no page reads (dead weight, not a bug):")
        print("\n".join(dead_rows))

    if absent:
        print(f"\n{absent} page(s) in PAGES do not exist on disk.")
    print("\nRESULT:", "all pages satisfied" if not (broken or absent)
          else f"{broken} page(s) broken, {absent} missing")
    return 1 if (broken or absent) else 0


if __name__ == "__main__":
    raise SystemExit(main())
