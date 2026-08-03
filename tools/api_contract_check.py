"""Every field a page reads must exist in the payload the server sends it.

    python tools/api_contract_check.py

Exits non-zero if any page reads a key the server never sends.

This is the failure mode the dashboard is most prone to, because it is silent:
the data shape changes, a page still reads the old key, and the panel renders
blank with nothing in any console to say why. Nobody notices until they go
looking for a number that used to be there.

It works by scraping `s.<field>` out of each page's script and checking it
against the real dict the corresponding server function returns - so it runs the
actual code rather than trusting a schema written alongside it.

Note the deliberate consequence: shadowing `s` inside a page (using it for a
series or a step rather than the state) makes this report false positives. That
is a feature. `s` means the page state in every one of these files, and a second
meaning for it is worth renaming.

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


def main() -> int:
    pages = {
        "index.html": server.summary_state(),
        "monitor.html": server.monitor_state(),
        "stage1.html": server.stage_state(1),
        "stage2.html": server.stage_state(2),
        "stage3.html": server.stage_state(3),
    }

    broken = 0
    for page, payload in pages.items():
        text = (HERE / page).read_text(encoding="utf-8")
        used = sorted(set(re.findall(r"\bs\.([A-Za-z_]\w*)", text)))
        missing = [f for f in used if f not in payload]
        if missing:
            broken += 1
        print(f"{page:<14} reads {len(used):>2} fields -> "
              f"{'MISSING ' + ', '.join(missing) if missing else 'ok'}")
        if missing:
            print(f"               {', '.join(used)}")

    print("\nkeys the server sends that no page reads (dead weight, not a bug):")
    for page, payload in pages.items():
        text = (HERE / page).read_text(encoding="utf-8")
        used = set(re.findall(r"\bs\.([A-Za-z_]\w*)", text))
        dead = sorted(set(payload) - used)
        if dead:
            print(f"  {page:<14} {', '.join(dead)}")

    print("\nRESULT:", "all pages satisfied" if not broken else f"{broken} page(s) broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
