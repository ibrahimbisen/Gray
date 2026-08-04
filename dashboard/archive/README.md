# The eight-page dashboard, archived 3 Aug 2026

Kept because these files are where the current six pages came from, and because a
diff against them is the only record of what moved where. Nothing serves them:
`server.py` resolves a URL to `dashboard/<name>.html`, and this is a subdirectory,
so `/stage2` is now a 404 rather than a page from before the rebuild.

## What replaced what

| archived | now | what happened |
|---|---|---|
| `overview.html` | `control.html` | kept the log-scale bar track, the three-number rule and the blocked-rows grouping. Gained the queue and job-log tabs. Lost its own run table - `/runs` has that, built better. |
| `monitor.html` | `runs.html` | every chart, tab and live-refresh mechanism carried over byte-for-byte. Lost 209 lines of its own CSS and its duplicated helpers. Gained a nav bar; it had none. |
| `index.html` | `summary.html` | content unchanged. Its six diagrams kept their private CSS, rewritten onto the shared tokens. Lost the "Pages" section, which the nav does now. |
| `pose.html` | `robot.html`, Pose tab | behaviour identical - same dials, same endpoints, same travel editor. Gained a nav bar, replacing a `← Monitor` link that pointed at a page which had stopped being the monitor. |
| `stage1.html` | `robot.html`, Model tab | every panel kept. |
| `stage3.html` | `robot.html`, Hardware tab | every panel kept. |
| `stage2.html` | `train.html` | kept whole. Renumbered to PLAN.md and the by-category table swapped for the collapse table, because the library it reads went from 200 rows to 60. |

`plan.html` is not here. It was rewritten in place, from the A-F phase structure
to PLAN.md's five steps.

## Why there were eight and now six

Not the count. The overlap:

- the **run table** was on two pages, rendered by two different bits of code
- the **plan** was on four pages, and they showed four *different* plans
- `/index` and `/index.html` served two different pages
- the pose editor had no way back to anything

And two stylesheets that both defined `--line`, `--ink`, `--warn` and `--accent`
with **different values**, so a component copied between pages silently changed
shade, and the same criterion printed as `2125.0`, `2125` and `2125.00` depending
on which page you were looking at.
