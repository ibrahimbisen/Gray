"""Every class a page uses must be defined, and every class defined should be used.

    python tools/css_check.py

Exits non-zero if a page uses a class nothing styles.

The companion to tools/api_contract_check.py. That one checks the data contract
between a page and its endpoint; this one checks the visual contract between a
page and the stylesheet. Both faults are silent in the same way: nothing appears
in any console, the page simply renders wrong, and it is only found by eye.

It caught its first one immediately. `.hide` and `.hidden` were two names for
`display:none`, and merging them into one left runs.html asking for a class that
no longer existed - so its metric charts would have stayed visible underneath the
table that replaced them.

WHAT IT CANNOT SEE. Class names built at runtime out of a variable - the
`cls()` helper in page.js returns "ok" / "bad" / "now", and `.dot ${k}` in
train.html interpolates a coverage word - are unknowable without running the
page. Those names are listed in BUILT below, by hand, and the list is short
because building a class name out of data is rare and worth keeping rare.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "dashboard"

# Class names assembled at runtime, which no amount of reading the source finds.
# Each is followed by the code that builds it.
BUILT = {
    # page.js cls(): a run state becomes a pill and a lamp modifier
    "ok", "bad", "warn", "now", "done", "doing", "failed", "running", "live",
    "muted", "pass", "fail", "none",
    # page.js srcChip(): where a number came from
    "src", "verified", "unlogged",
    # the coverage words, from skills.py, used as `.dot ${k}` and `.cov i.${k}`
    "command", "condition", "emerges", "pilot", "measured", "blocked",
    "sel", "on", "hidden", "drift", "y", "n",
}

# Classes that are hooks, not styles: page.js finds them with querySelector, or
# the element is positioned by an inline style attribute and the class is only
# there to say what it is. The stylesheet is right not to have a rule for these.
HOOKS = {
    "pane",          # tabbed() puts the drawn tab here
    "hit", "cross",  # the chart's hover target and crosshair, both inline-sized
}
BUILT |= HOOKS

# Attributes whose value is a class list, and the template forms of the same.
CLASS_ATTR = re.compile(r'class\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|`([^`]*)`)')
CLASSLIST = re.compile(r'classList\.(?:add|remove|toggle)\(\s*"([^"]+)"')
# `className = "banner bad"`
CLASSNAME = re.compile(r'className\s*=\s*"([^"]*)"')
# A selector in the stylesheet: .foo, .foo:hover, .foo > h2 ...
DEFINED = re.compile(r"\.([A-Za-z][\w-]*)")
# Strip CSS comments before scraping, or every class named in a comment counts.
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
WORD = re.compile(r"[A-Za-z][\w-]*")


def _strip_interp(text: str) -> str:
    """Remove every ${...} from a template literal, counting braces.

    Needed before any class attribute can be read. `class="rc ${r.id === SELECTED
    ? "sel" : ""}"` contains quotes INSIDE the interpolation, so the attribute
    regex stops at the first one and hands back `rc ${r.id === SELECTED ? `. Read
    naively that says the page uses a class called SELECTED. A regex cannot count
    braces, so this does it by hand, and it handles a template nested in an
    interpolation nested in a template - which this dashboard does have.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == "$" and i + 1 < n and text[i + 1] == "{":
            depth, i = 1, i + 2
            while i < n and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def used_in(text: str) -> set[str]:
    """Classes the page definitely asks for. Strict: a claim of breakage."""
    flat = _strip_interp(text)
    out: set[str] = set()
    for match in CLASS_ATTR.finditer(flat):
        value = next(g for g in match.groups() if g is not None)
        out.update(w for w in value.split() if WORD.fullmatch(w))
    for pattern in (CLASSLIST, CLASSNAME):
        for match in pattern.finditer(flat):
            out.update(w for w in match.group(1).split() if w)
    return out


def mentioned_in(text: str) -> set[str]:
    """Every word in every quoted string. Loose: used only to judge what is dead.

    Deliberately over-broad. A class assembled inside an interpolation - `${v.driven
    ? " driven" : ""}` - is invisible to used_in above, and reporting a live class
    as dead is how a stylesheet loses a rule that something needed.
    """
    out: set[str] = set()
    for match in re.finditer(r'"([^"\n]*)"|\'([^\'\n]*)\'', text):
        value = next(g for g in match.groups() if g is not None)
        out.update(w for w in value.split() if WORD.fullmatch(w))
    # And every single quoted word anywhere, whatever it is nested inside. The
    # pass through above walks quotes in order, so it mis-pairs them across an
    # interpolation - `class="${p ? "pos" : "neg"}"` hid both of those and the
    # report called two live classes dead.
    out.update(re.findall(r'"([A-Za-z][\w-]*)"', text))
    return out


def defined_in(css: str) -> set[str]:
    return set(DEFINED.findall(CSS_COMMENT.sub("", css)))


TOP_LEVEL = re.compile(r"^(?:const|let|function)\s+([A-Za-z_$][\w$]*)", re.M)


def redeclared(page_js: str, page_text: str) -> list[str]:
    """Names declared at the top level of BOTH page.js and a page.

    This is a SyntaxError, not a shadow. Two classic scripts share one global
    lexical scope, so the second `const blk = ...` stops the whole page script
    from parsing. The page then loads, the stylesheet applies, and nothing runs -
    it sits on "Loading…" for ever, and the only trace is in a console nobody has
    open.

    It has happened twice: `trackPos` when the on-track card moved into page.js,
    and `blk` when the block component did. Both times the symptom was a blank
    page and both times it cost an hour. A grep is cheaper.
    """
    # Only the page's own <script> blocks; markup cannot declare anything.
    scripts = "\n".join(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
                                   page_text, re.S))
    return sorted(set(TOP_LEVEL.findall(page_js)) & set(TOP_LEVEL.findall(scripts)))


def main() -> int:
    shared = (HERE / "page.css").read_text(encoding="utf-8")
    shared_classes = defined_in(shared)
    page_js = (HERE / "page.js").read_text(encoding="utf-8")

    pages = sorted(p for p in HERE.glob("*.html"))
    missing_total = 0
    all_seen: set[str] = mentioned_in(page_js)

    for page in pages:
        text = page.read_text(encoding="utf-8")
        own = "\n".join(re.findall(r"<style>(.*?)</style>", text, re.S))
        known = shared_classes | defined_in(own) | BUILT
        used = used_in(text)
        all_seen |= mentioned_in(text)
        missing = sorted(used - known)
        if missing:
            missing_total += len(missing)
        clash = redeclared(page_js, text)
        if clash:
            missing_total += len(clash)
        note = ("UNDEFINED " + ", ".join(missing)) if missing else "ok"
        if clash:
            note = f"REDECLARES {', '.join(clash)} from page.js - SyntaxError"
        print(f"{page.name:<17} {len(used):>3} classes, "
              f"{len(defined_in(own)):>3} of its own -> {note}")

    # page.js writes markup for every page, so its classes are checked against
    # the shared sheet only - it cannot rely on any one page's local styles.
    js_missing = sorted(used_in(page_js) - shared_classes - BUILT)
    print(f"{'page.js':<17} {len(used_in(page_js)):>3} classes"
          f"{'':>19}-> {'UNDEFINED ' + ', '.join(js_missing) if js_missing else 'ok'}")
    missing_total += len(js_missing)

    dead = sorted(shared_classes - all_seen - BUILT)
    if dead:
        print(f"\ndefined in page.css, used by nothing ({len(dead)}):")
        print("  " + ", ".join(dead))

    print("\nRESULT:", "every class is defined" if not missing_total
          else f"{missing_total} undefined class(es)")
    return 1 if missing_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
