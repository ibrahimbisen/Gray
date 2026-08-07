/* Shared by all six pages. Every helper in here existed two to five times over
   before, in versions that disagreed - which is why the same criterion printed as
   `2125.0` on one page, `2125` on another and `2125.00` on a third. There is one
   of each now, and pages import rather than re-implement.

   Loaded as a plain script, so everything here is a global on purpose. */

/* ===================================================================== text === */

const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const $ = (s, root) => (root || document).querySelector(s);

/* The one number formatter. Magnitude-adaptive, because a reward of 0.03 and a
   step count of 24,576 cannot share a decimal place.

   It takes non-numbers on purpose: metrics.csv can hold a text cell - a header
   repeated by a crashed writer - and runs.py keeps it as a string rather than
   inventing a zero. `v.toFixed` would throw and take the whole panel down. */
function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v !== "number" || !isFinite(v)) return esc(String(v));
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString();
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/* A criterion carries its own unit, so the unit picks the precision rather than
   each page guessing. Sideways drift in mm and tracking error in m/s do not want
   the same number of decimals, and neither wants `fmt`'s magnitude rule. */
function value(v, unit) {
  if (v === null || v === undefined) return "—";
  if (unit === "fraction") return (v * 100).toFixed(0) + "%";
  if (unit === "ratio")    return (+v).toFixed(4);
  if (unit === "mm")       return (+v).toFixed(1) + " mm";
  if (unit === "m")        return (+v).toFixed(2) + " m";
  if (unit === "m/s")      return (+v).toFixed(3) + " m/s";
  if (unit === "deg")      return (+v).toFixed(1) + "°";
  if (unit === "s")        return (+v).toFixed(1) + " s";
  /* rad/s is the unit of `turn_err`, and it had no case here. Every criterion
     that carries it fell through to the line below and printed the raw double -
     `turn rate 0.11773093044757843` sat in a column of numbers rounded to three
     places, on four pages at once. Three decimals, to match m/s: both are a
     rate, and the bar for this one is 0.2. */
  if (unit === "rad/s")    return (+v).toFixed(3) + " rad/s";
  /* Anything else is a unit nobody has taught this function. `fmt` at least
     rounds it, which is a better answer than seventeen significant figures - but
     the unit still needs a case above, so it says so. */
  if (typeof v === "number") {
    console.warn(`value(): no rule for unit ${JSON.stringify(unit)}`);
    return fmt(v) + (unit ? " " + unit : "");
  }
  return esc(String(v));
}

/* Seconds as a human duration. The server has its own copy for durations it
   writes into the payload; this one is for values computed in the browser, which
   is only ever an ETA. */
function clock(secs) {
  if (!isFinite(secs) || secs < 0) return "—";
  const h = Math.floor(secs / 3600), m = Math.round((secs % 3600) / 60);
  return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${Math.max(1, m)} min`;
}

const label = (k) => String(k).replace(/_/g, " ").replace(/\bmm\b/, "(mm)");

/* ================================================================== status === */

/* Every state string the server can produce, mapped to one of five looks. This
   was written four times with four different sets of keys, so a run that
   "reached target" showed green on two pages and unstyled grey on a third.
   An unknown state deliberately falls to `muted` rather than to a colour - a
   guessed colour is worse than no colour. */
const STATE_CLASS = {
  "running": "now", "reached target": "ok", "finished": "ok",
  "cancelled": "warn", "interrupted": "warn", "failed": "bad",
  "passed": "ok", "not passed": "bad", "partly": "warn",
  "queued": "warn", "done": "ok", "skipped": "warn",
  "in progress": "now", "not started": "muted", "ongoing": "now",
  "parked": "muted", "shelved": "muted", "blocked": "bad",
  "should be running": "warn", "provisional": "warn",
};
const cls = (s) => STATE_CLASS[String(s || "").toLowerCase()] || "muted";

const chip = (s) => `<span class="chip ${cls(s)}">${esc(s)}</span>`;

/* ratio is 1.0 at the bar and above 1.0 passing, in both directions. So a failing
   criterion is 1/ratio times over its bar. */
const timesOver = (c) => (!c.ratio || c.ratio <= 0 || c.ratio >= 1) ? null : 1 / c.ratio;

/* ===================================================================== nav === */

/* THE list of pages. dashboard/server.py parses this array to print its startup
   banner, so the terminal and this bar cannot disagree. They did: the banner
   named seven, this named eight, and server.py's docstring said six. */
const PAGES = [
  ["/",        "Now",     "what is happening right now"],
  ["/runs",    "Runs",    "how did that run go"],
  ["/plan",    "Plan",    "what we are doing next, and why"],
  ["/train",   "Train",   "what it is being taught"],
  ["/dials",   "Dials",   "every number that gets varied, and what it was"],
  ["/config",  "Config",  "every number that can be changed, and where it lives"],
  ["/robot",   "Robot",   "is the physical model right"],
  ["/controller", "Controller", "what each control does, and what is still free"],
  ["/summary", "Summary", "how the whole thing works"],
];

/* One bar, on every page. `right` is the live status string - the current plan
   step, what is training, and the worst criterion - computed server-side once so
   all six pages agree instead of three of them recomputing it. */
function topbar(active, right) {
  const here = active || location.pathname;
  const link = ([href, label, title]) =>
    `<a href="${href}" title="${esc(title)}" class="${href === here ? "on" : ""}">${label}</a>`;
  return `<div class="top">
    <div class="brand">Gray</div>
    ${PAGES.map(link).join("")}
    <div class="sp"></div>
    <div class="state">${right || ""}</div>
  </div>`;
}

/* Put the bar at the top of the document, above whatever the page's own root is.
   Every page calls this once. */
function mountNav(active, right) {
  const el = document.getElementById("nav");
  if (el) el.outerHTML = topbar(active, right);
  else document.body.insertAdjacentHTML("afterbegin", topbar(active, right));
}

/* The status string itself, from the payload every endpoint carries. */
function navState(s) {
  if (!s) return "";
  const bits = [];
  if (s.step) bits.push(`<span>${esc(s.step)}</span>`);
  if (s.training) bits.push(`<span><span class="dot live"></span>${esc(s.training)}</span>`);
  else bits.push(`<span><span class="dot"></span>nothing training</span>`);
  if (s.worst) bits.push(`<span>${esc(s.worst)}</span>`);
  return bits.join("");
}

/* ================================================================== paper === */

/* One continuous page, laid out like a report: numbered sections, a contents rail
   on the left. Used where the point is to read end to end and hold two tables
   against each other.

   sections: [{id, label, count, lead, html}] - lead:true is the unnumbered
   opening block. `html` may be a string or a function. */
function paper(sections, opts = {}) {
  const side = document.querySelector(".side");
  const main = document.querySelector(".main") || document.querySelector("main");

  let n = 0;
  const secs = sections.map(s => ({ ...s, num: s.lead ? null : ++n }));

  if (side) {
    side.innerHTML = `<h4>${esc(opts.contents || "Contents")}</h4>` + secs.map(s =>
      `<button data-id="${esc(s.id)}">${
        s.num ? `<span class="id">${s.num}</span>` : ""
      }${esc(s.label)}${
        s.count != null ? `<span class="n">${s.count}</span>` : ""
      }</button>`).join("");
  }

  main.innerHTML = `<div class="paper">` + secs.map(s => `
    <section id="${esc(s.id)}"${s.lead ? ` class="lead"` : ""}>
      ${s.num ? `<h2><span class="secnum">${s.num}</span>${esc(s.label)}</h2>` : ""}
      ${typeof s.html === "function" ? s.html() : s.html}
    </section>`).join("") + `</div>`;

  if (!side) return;
  const buttons = new Map(
    [...side.querySelectorAll("button[data-id]")].map(b => [b.dataset.id, b]));
  const mark = (id) => buttons.forEach((b, k) => b.classList.toggle("on", k === id));
  const go = (id) => {
    const el = main.querySelector("#" + CSS.escape(id));
    if (el) el.scrollIntoView({ block: "start" });
  };

  side.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) { go(b.dataset.id); mark(b.dataset.id); }
  });

  // The contents rail follows the reader rather than the reader having to
  // remember where they got to. rootMargin pins "current" to the top band, so a
  // tall section stays selected the whole way down it.
  const onScreen = new Set();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) onScreen.add(e.target.id);
      else onScreen.delete(e.target.id);
    }
    const first = secs.find(s => onScreen.has(s.id));
    if (first) {
      mark(first.id);
      if (location.hash.slice(1) !== first.id)
        history.replaceState(null, "", "#" + first.id);
    }
  }, { rootMargin: "0px 0px -72% 0px" });
  main.querySelectorAll("section[id]").forEach(el => io.observe(el));

  const start = location.hash.slice(1);
  if (buttons.has(start)) go(start);
  mark(buttons.has(start) ? start : secs[0].id);
}

/* ================================================================== tabs ==== */

/* A row of tabs over one pane. `tabs` is [{id, label, count, html}]. The open tab
   goes in the URL hash, so it survives a refresh and can be linked to. */
function tabbed(root, tabs, opts = {}) {
  const el = typeof root === "string" ? document.querySelector(root) : root;
  const draw = (id) => {
    const t = tabs.find(x => x.id === id) || tabs[0];
    el.innerHTML = `<div class="tabs">${tabs.map(x =>
      `<button class="tab ${x.id === t.id ? "on" : ""}" data-t="${esc(x.id)}">${esc(x.label)}${
        x.count != null ? `<span class="n">${x.count}</span>` : ""}</button>`).join("")}</div>
      <div class="pane">${typeof t.html === "function" ? t.html() : t.html}</div>`;
    el.querySelectorAll("button[data-t]").forEach(b =>
      b.onclick = () => { history.replaceState(null, "", "#" + b.dataset.t); draw(b.dataset.t); });
    if (opts.onDraw) opts.onDraw(t.id, el.querySelector(".pane"));
  };
  const start = location.hash.slice(1);
  draw(tabs.some(t => t.id === start) ? start : tabs[0].id);
  window.addEventListener("hashchange", () => draw(location.hash.slice(1)));
}

/* =============================================================== builders === */

/* THE block: a bordered panel with a mono caption bar that names it and counts
   what is in it. Promoted out of carry.html, which is where the dashboard's
   density came from. Every page builds these, so the markup is written once.

   `count` is optional and `right` takes anything that belongs on the far end of
   the caption - a chip, a link. The body is raw HTML, because it is always a
   table or a grid the caller has already built. */
const blk = (title, count, body, right) => `
  <div class="blk"><h2><b>${esc(title)}</b>${
    count != null ? `<span class="n">${count}</span>` : ""}${
    right ? `<span class="n">${right}</span>` : ""}</h2>${body}</div>`;

/* The title block at the head of a sheet. `cells` is [label, value] pairs, and
   the first one is the sheet's name and gets the big treatment.

   Every page states where its numbers come from, here, in the same place, in the
   same words. That is the rule this repo runs on: a number without a source is
   the fault verify.py warns about. Pass `{html:true}` on a cell whose value is
   markup, otherwise it is escaped. */
const titleblock = (cells) => `<div class="titleblock">${cells.map((c, i) => `
  <div class="${i === 0 ? "name" : ""}">
    <div class="k">${esc(c.k)}</div>
    <div class="v ${i === 0 ? "big" : ""}">${c.html || esc(c.v ?? "")}</div>
  </div>`).join("")}</div>`;

/* The first sentence of an explanation, for a `why` column beside a number.

   A reward term's note runs to five or six sentences, which is right on the page
   that explains the reward function and wrong in a table cell - twenty-four of
   them turn a table into an essay. The whole note goes on the row's `title`, so
   nothing is lost, and /summary is one link away. */
const firstSentence = (s) => {
  const t = String(s || "").trim();
  const cut = t.search(/\.\s/);
  return cut === -1 ? t : t.slice(0, cut + 1);
};

const facts = (rows) => `<div class="facts">${rows.map(f => `
  <div class="fact"><div class="k">${esc(f.k)}</div>
    <div class="v">${f.html || esc(f.v)}</div>
    <div class="n">${esc(f.n ?? f.note ?? "")}</div></div>`).join("")}</div>`;

const tiles = (rows) => `<div class="tiles">${rows.map(t => `
  <div class="tile"><div class="k">${
      t.swatch ? `<span class="swatch" style="background:${t.swatch}"></span>` : ""
    }${esc(t.k)}</div>
    <div class="v">${t.html || esc(t.v)}</div>
    <div class="d">${t.dHtml || esc(t.d ?? "")}</div></div>`).join("")}</div>`;

function modelBlock(m) {
  if (!m || !m.exists) return `<p class="muted">No model yet.</p>`;
  if (m.error) return `<p class="muted">Could not check: <code>${esc(m.error)}</code></p>`;
  const pct = m.total ? Math.round(100 * m.passed / m.total) : 0;
  const col = m.passed === m.total ? "var(--good)"
            : m.passed > m.total / 2 ? "var(--warn)" : "var(--crit)";
  return `
    <div class="status">
      <div class="score" style="color:${col}">${m.passed} / ${m.total}</div>
      <div class="bar"><i style="width:${pct}%;background:${col}"></i></div>
    </div>
    <div class="muted small" style="margin-bottom:6px"><code>${esc(m.urdf)}</code>
      &mdash; re-run every time this page loads</div>
    ${m.checks.map(c => `
      <div class="check">
        <div class="mark ${c.passed ? "y" : "n"}">${c.passed ? "&#10003;" : "&#10007;"}</div>
        <div><b>${esc(c.title)}</b>
          <div class="detail">${esc(c.detail || "")}</div>
          ${(c.rows || []).length
            ? `<ul>${c.rows.map(p => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
        </div></div>`).join("")}`;
}

/* ========================================================== the on-track card === */

/* "Is this run going to pass the bar?" - the same six gates scripts/verify.py
   scores, one row each. Built here rather than on a page because BOTH pages ask
   it: the Control page about whatever is training, and the Runs page about
   whichever run you have open. It was written on Control first and was about to
   be copied; `chartCard` above is in this file for exactly the same reason.

   Every row carries where its number came from, because the two sources are not
   the same kind of fact:

     verified        scripts/verify.py measured it on a checkpoint, on a fixed
                     test, after the run. This is the real thing.
     from training   read off metrics.csv as the run goes. Measures something
                     ADJACENT to the bar - see the caveat on each chip - and is
                     indicative, never a verdict.
     not logged      nothing measures it during training. Says so, borrows
                     nothing, and shows the last verified number instead.

   A run that has been verified shows its OWN verdict; one still training shows
   estimates. The card is the same either way, which is the point - you read the
   two against each other without re-learning the layout. */

/* Position on the distance axis. Log scale, fixed 0.1x - 100x domain, so the bar
   tick sits at the same x in every row and rows are comparable at a glance. A
   linear "percent of bar" would put something at 4% of target near a full-looking
   bar's left edge, which is the lie this exists to avoid. */
const trackPos = (r) => (!r || r <= 0) ? null
  : Math.max(0, Math.min(100, ((Math.log10(r) + 1) / 3) * 100));

/* Distance to bar, in words. `ratio` is 1.0 AT the bar and above 1.0 passing, in
   both directions - so the multiple reads the same whether the criterion wants
   more or less. */
function gapText(ratio, passed) {
  if (ratio === null || ratio === undefined || !isFinite(ratio) || ratio <= 0) return "";
  return passed === false ? `${fmt(1 / ratio)}&times; over`
       : passed === true  ? `${fmt(ratio)}&times; inside`
       : `${ratio >= 1 ? fmt(ratio) + "&times; inside" : fmt(1 / ratio) + "&times; over"}`;
}

/* Two readings of one criterion in one slot: the lamp is the number this row is
   reporting, the hollow ring is the other one - so a live estimate is shown
   against the verifier's last word, and a verdict against what training thought
   at the time. */
function boardTrack(r) {
  const now = trackPos(r.ratio);
  const was = trackPos(r.other && r.other.ratio);
  if (now === null && was === null) return `<span class="faint small">&mdash;</span>`;
  const lamp = now === null ? ""
    : `<span class="dot ${r.passed === true ? "ok" : r.passed === false ? "bad" : ""}"
         style="left:${now}%" title="${esc(r.source === "verified"
           ? "measured by verify.py" : "estimated from training")}"></span>`;
  const ghost = was === null ? ""
    : `<span class="ghost" style="left:${was}%" title="${
        esc(r.other.what || "the other reading")}"></span>`;
  return `<div class="track"><div class="ax"></div>
    <div class="tick" style="left:33.3%"></div>${ghost}${lamp}</div>`;
}

/* Where a number came from, as a word. `.badge` already IS this pill - see the
   note beside `.badge.src` in page.css - so this is a modifier of it and not a
   fourth spelling. Never the word "live": a card can be looking at a run that
   finished last week, and every row labelled LIVE would be the page lying about
   its own freshness. */
function srcChip(source, caveat) {
  const word = { live: "from training", verified: "verified",
                 unmeasured: "not logged" }[source] || esc(source);
  const mod = source === "verified" ? "verified"
            : source === "unmeasured" ? "unlogged" : "";
  // How this number differs from what the bar asks for hangs off the chip that
  // names where it came from, which is the thing you would hover to ask. On the
  // row itself it was a line of grey text on every passing gate - noise on the
  // rows that are fine, and it buried the ones that are not.
  return `<span class="badge src ${mod}"${caveat ? ` title="${esc(caveat)}"` : ""
    }>${word}</span>`;
}

/* The legend. The scale is built on `.lrow` rather than a div of its own so it
   borrows the row grid: `0.1x`, `bar` and `100x` land over the track column
   exactly, and stay there when the window changes width. A hand-placed margin
   would be right at one width only. */
function boardLegend() {
  return `<div class="legend marks">
      <span><b style="color:var(--good)">✓</b>met</span>
      <span><b style="color:var(--crit)">✗</b>failing</span>
      <span><b>·</b>not judged</span>
      <span><span class="dot ok"></span>this row's number</span>
      <span><span class="ghostkey"></span>the other reading</span>
    </div>
    <div class="lrow head">
      <span></span><span></span><span></span><span></span>
      <div class="scalerow"><span>0.1&times;</span><span class="at">bar</span>
        <span>100&times;</span></div>
      <span class="gap">distance</span>
      <span class="lnote">note</span>
    </div>`;
}

/* The LAST CELL of a row, not a line under it. It was a `.lsub` block once, and
   three of the six gates had a paragraph wedged beneath them - so the board read
   as three separate huddles instead of one set of gates. The gates are the whole
   point; they have to sit together. Anything that will not fit on the line lives
   on the `title` and in a footnote under the whole board. */
function boardNote(row) {
  if (row.source === "unmeasured") {
    const text = row.last
      ? `not logged &mdash; last verified <b>${value(row.last.value, row.unit)}</b> on
         <a href="/runs#run=${encodeURIComponent(row.last.run)}">${
         esc(row.last.label)}</a>${row.last.ratio && row.last.ratio < 1
           ? `, ${fmt(1 / row.last.ratio)}&times; over` : ""}`
      : `not logged while training, and never verified either`;
    return `<span class="lnote" title="nothing logs this criterion during training">${
      text}</span>`;
  }
  if (row.source === "verified" && row.note)
    return `<span class="lnote" title="${esc(row.note)}">${esc(row.note)}</span>`;
  if (!row.judged && row.why_unjudged)
    return `<span class="lnote" title="${esc(row.why_unjudged)}">reported, not judged`
         + ` &mdash; see below</span>`;
  // A gate that is measured and judged says nothing here. Its caveat is on the
  // source chip, where hovering "from training" asks exactly that question.
  return `<span class="lnote"></span>`;
}

/* The long caveats, gathered UNDER the board. One line each, named by the
   criterion it belongs to, so it can be tied back without the prose sitting
   inside the rows and breaking them apart. */
function boardFootnotes(rows) {
  const out = rows.filter(r => !r.judged && r.why_unjudged && r.source !== "unmeasured");
  if (!out.length) return "";
  return out.map(r => `<div class="lsub" style="margin-top:7px">
    <b>${esc(r.name)} is reported, not judged.</b> ${esc(r.why_unjudged)}</div>`).join("");
}

/* `board` is dashboard/live.py's bar_board(); `run` is the run it describes.
   opts.training says the run is still going, which only changes the subtitle. */
function onTrackCard(board, run, opts) {
  const o = opts || {}, r = run || {};
  if (!board || !board.has_verifier) return `<div class="readout"><h3>On track?</h3>
    ${empty(`${String(r.task || "this task").replace("Gray-", "")} has no verifier,
             so there is no bar to be on track for.`)}</div>`;

  const sub = o.training
    ? `${(r.iterations_done || 0).toLocaleString()}/${(r.iterations_target || 0)
        .toLocaleString()} &middot; ${Math.round(100 * (r.progress || 0))}%`
    : `${esc(r.status || "")}${r.duration ? ` &middot; ${esc(r.duration)}` : ""}`;
  const anyLive = board.rows.some(x => x.source === "live");

  return `<div class="readout">
    <h3>On track?
      <span class="who">${r.number ? "#" + r.number + " " : ""}${
        esc(r.variant || r.name || "")}</span>
      <span class="say">${sub}</span>
      <span class="grow"></span>
      <span class="say"><b style="color:var(--good)">${board.met} met</b>
        &middot; <b style="color:${board.failing ? "var(--crit)" : "var(--muted)"}">${
          board.failing} failing</b>
        &middot; ${board.unknown} not judged</span></h3>
    ${boardLegend()}
    ${board.rows.map(row => {
      const cls = row.passed === true ? "pass" : row.passed === false ? "fail" : "";
      const mk = row.passed === true ? "✓" : row.passed === false ? "✗" : "·";
      const has = row.measured !== null && row.measured !== undefined;
      return `<div class="lrow ${cls}">
        <span class="mk">${mk}</span>
        <span class="nm">${esc(row.name)}</span>
        <span class="v ${has ? "" : "none"}">${
          has ? value(row.measured, row.unit) : "&mdash;"}</span>
        ${srcChip(row.source, row.caveat)}
        ${boardTrack(row)}
        <span class="gap">${row.judged ? gapText(row.ratio, row.passed)
          : "bar " + value(row.bar, row.unit)}</span>
        ${boardNote(row)}
      </div>`;
    }).join("")}
    ${boardFootnotes(board.rows)}
    ${!anyLive ? "" : `<div class="lsub" style="padding-left:0;margin-top:7px">
      <b>A number off the training logs is not a verdict.</b> It measures something
      next to what the bar asks for; <code>verify.py</code> scores a checkpoint on a
      fixed test after the run, and that is the one that counts.
    </div>`}
  </div>`;
}

/* ================================================================== charts === */

/* One line chart, used by /runs (forty-four of them, 132px) and by / (four of
   them, 96px, one of which carries a reference line). It lived on /runs and was
   about to be copied onto the control page, which is how this file got its
   docstring - so it moved here first.

   Never two y-scales on one plot: the alignment between them is arbitrary and
   invents a correlation that is not in the data. One metric, one chart. */

const CHART = { W: 480, H: 132, L: 44, R: 10, T: 8, B: 22 };

/* The filter, in ONE place, because the chart and its crosshair both need it and
   a disagreement between them lands the crosshair on a different sample than the
   line it is tracking.

   `+null` and `+""` are both 0, so mapping every row plotted a fabricated zero
   wherever a value was missing: a metric that starts logging at iteration 50 drew
   a flat zero line before it, dragging the y-axis down and squashing the real
   curve. Worse, a NaN reward showed as "0.00" on the chart while the tile and the
   table both said "—" - three numbers for one cell, and the chart's was invented. */
function chartPoints(rows, key) {
  const pts = [];
  for (const r of rows || []) {
    const v = r[key];
    if (typeof v === "number" && isFinite(v))
      pts.push({ x: r.iteration ?? pts.length, y: v });
  }
  return pts;
}

/* The y-range includes `extra` - the reference line - so a bar sitting outside
   the data's own range is still on the canvas rather than clipped off the top. */
function chartScales(xs, ys, geo, extra) {
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const all = (typeof extra === "number" && isFinite(extra)) ? ys.concat([extra]) : ys;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { hi = lo + 1; lo -= 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  return {
    x0, x1, lo, hi,
    px: v => geo.L + (geo.W - geo.L - geo.R) * ((v - x0) / ((x1 - x0) || 1)),
    py: v => geo.T + (geo.H - geo.T - geo.B) * (1 - (v - lo) / ((hi - lo) || 1)),
  };
}

/* Gridlines and axes are SOLID hairlines one shade off the surface. A dashed
   grid reads as "threshold" or "projection" when it is only a grid. */
function chartAxes(sc, geo) {
  const ticksY = [sc.lo + (sc.hi - sc.lo) * 0.08, (sc.lo + sc.hi) / 2,
                  sc.hi - (sc.hi - sc.lo) * 0.08];
  const ticksX = [sc.x0, (sc.x0 + sc.x1) / 2, sc.x1];
  return `
    ${ticksY.map(t => `<line x1="${geo.L}" x2="${geo.W - geo.R}" y1="${sc.py(t).toFixed(1)}"
       y2="${sc.py(t).toFixed(1)}" stroke="var(--grid)" stroke-width="1"
       vector-effect="non-scaling-stroke"/>`).join("")}
    <line x1="${geo.L}" x2="${geo.L}" y1="${geo.T}" y2="${geo.H - geo.B}" stroke="var(--line)"
          stroke-width="1" vector-effect="non-scaling-stroke"/>
    <line x1="${geo.L}" x2="${geo.W - geo.R}" y1="${geo.H - geo.B}" y2="${geo.H - geo.B}"
          stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>
    ${ticksY.map(t => `<text x="${geo.L - 6}" y="${(sc.py(t) + 3.5).toFixed(1)}" text-anchor="end"
       font-size="9" fill="var(--muted)" class="tnum">${fmt(t)}</text>`).join("")}
    ${ticksX.map((t, i) => `<text x="${sc.px(t).toFixed(1)}" y="${geo.H - geo.B + 13}"
       text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}"
       font-size="9" fill="var(--muted)" class="tnum">${Math.round(t)}</text>`).join("")}`;
}

/* opts: {colour, geo, bar, barLabel, note, range}
   `bar` draws a reference hairline. `range` prints the min-to-max line under the
   title - /runs wants it, the control page has no room for it. */
function chartCard(rows, key, opts) {
  const o = opts || {};
  const geo = o.geo || CHART;
  const colour = o.colour || "var(--s1)";
  const pts = chartPoints(rows, key);
  const head = (now, tone) => `<h3><span class="swatch" style="background:${colour}"></span>${
    esc(label(key))}<span class="now" style="color:${tone}">${now}</span></h3>`;

  if (pts.length < 2) return `<div class="card">${head("—", "var(--muted)")}
    <div class="rng">only ${pts.length} usable sample${pts.length === 1 ? "" : "s"}
      out of ${(rows || []).length} rows</div></div>`;

  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const sc = chartScales(xs, ys, geo, o.bar);
  const d = pts.map((p, i) =>
    `${i ? "L" : "M"}${sc.px(p.x).toFixed(1)},${sc.py(p.y).toFixed(1)}`).join("");
  const endX = sc.px(xs[xs.length - 1]), endY = sc.py(ys[ys.length - 1]);
  const gaps = (rows || []).length - pts.length;
  const barY = (typeof o.bar === "number" && isFinite(o.bar)) ? sc.py(o.bar) : null;

  return `<div class="card" data-key="${esc(key)}" data-geo="${geo.H}">
    ${head(fmt(ys[ys.length - 1]), colour)}
    ${o.note ? `<div class="rng">${esc(o.note)}</div>`
     : o.range === false ? ""
     : `<div class="rng tnum">${fmt(Math.min(...ys))} to ${fmt(Math.max(...ys))} over
        ${pts.length} samples${gaps ? ` &middot; ${gaps} row(s) had no value` : ""}</div>`}
    <div class="plot">
      <svg viewBox="0 0 ${geo.W} ${geo.H}" preserveAspectRatio="none" role="img"
           style="height:${geo.H}px"
           aria-label="${esc(label(key))} against training iteration">
        ${chartAxes(sc, geo)}
        ${barY === null ? "" : `<line class="barref" x1="${geo.L}" x2="${geo.W - geo.R}"
           y1="${barY.toFixed(1)}" y2="${barY.toFixed(1)}"
           vector-effect="non-scaling-stroke"/>
          <text x="${geo.W - geo.R}" y="${(barY - 4).toFixed(1)}" text-anchor="end"
             font-size="9" fill="var(--muted)">${esc(o.barLabel || "bar")}</text>`}
        <path d="${d}" fill="none" stroke="${colour}" stroke-width="2"
              stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
        <circle cx="${endX.toFixed(1)}" cy="${endY.toFixed(1)}" r="3.2" fill="${colour}"
                stroke="var(--surface)" stroke-width="2"/>
        <g class="cross" opacity="0">
          <line y1="${geo.T}" y2="${geo.H - geo.B}" stroke="var(--muted)" stroke-width="1"
                vector-effect="non-scaling-stroke"/>
          <circle r="4" fill="${colour}" stroke="var(--surface)" stroke-width="2"/>
        </g>
        <rect x="${geo.L}" y="0" width="${geo.W - geo.L - geo.R}" height="${geo.H}"
              fill="transparent" class="hit"/>
      </svg>
      <div class="tip"></div>
    </div>
  </div>`;
}

/* The crosshair. Snaps to the nearest sample in x, so the reader aims at an
   iteration rather than at a 2px line. */
function wireCharts(root, rows, opts) {
  const geo = (opts || {}).geo || CHART;
  (root || document).querySelectorAll(".card[data-key]").forEach(card => {
    const key = card.dataset.key, svg = card.querySelector("svg");
    const hit = card.querySelector(".hit"), g = card.querySelector(".cross");
    if (!hit || !g) return;
    const line = g.querySelector("line"), dotEl = g.querySelector("circle");
    const tip = card.querySelector(".tip");
    const pts = chartPoints(rows, key);
    if (pts.length < 2) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const sc = chartScales(xs, ys, geo, (opts || {}).bars ? (opts.bars[key]) : undefined);

    hit.addEventListener("mousemove", ev => {
      const b = svg.getBoundingClientRect();
      const vx = ((ev.clientX - b.left) / b.width) * geo.W;
      let best = 0, bd = Infinity;
      xs.forEach((x, i) => { const dd = Math.abs(sc.px(x) - vx); if (dd < bd) { bd = dd; best = i; } });
      const cx = sc.px(xs[best]), cy = sc.py(ys[best]);
      g.setAttribute("opacity", "1");
      line.setAttribute("x1", cx); line.setAttribute("x2", cx);
      dotEl.setAttribute("cx", cx); dotEl.setAttribute("cy", cy);
      tip.style.opacity = "1";
      tip.style.left = (cx / geo.W * 100) + "%";
      tip.style.top = (cy / geo.H * b.height) + "px";
      tip.innerHTML = `<b>${fmt(ys[best])}</b> &middot; iteration
        <span class="tnum">${xs[best]}</span>`;
    });
    hit.addEventListener("mouseleave", () => {
      g.setAttribute("opacity", "0"); tip.style.opacity = "0";
    });
  });
}

const empty = (msg) => `<div class="empty">${esc(msg)}</div>`;

/* One error state. Four pages each had their own, one of which left stale content
   visible underneath the message - so the page looked like it had loaded. */
function fail(err, where) {
  const el = document.querySelector(where || ".main") || document.querySelector("main");
  if (el) el.innerHTML = `<div class="paper"><h2>Could not load</h2>
    <p class="muted">${esc(err)}</p>
    <p class="muted small">If the dashboard was restarted, reload the page.
      If this keeps happening, the terminal running <code>run.bat</code> has the
      real error.</p></div>`;
}

/* One fetch wrapper, so a failed load never leaves a page half-drawn. */
async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.json();
}

/* POST, then reload. `alert` is the only interruption in the whole dashboard and
   it is deliberate: a queue edit that silently failed is worse than a dialog. */
let BUSY = false;
async function post(path, body, after) {
  if (BUSY) return null;
  BUSY = true;
  try {
    const r = await fetch(path, { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}) });
    const out = await r.json();
    if (out && out.error) alert(out.error);
    return out;
  } catch (e) { alert(String(e)); return null; }
  finally { BUSY = false; if (after) await after(); }
}
