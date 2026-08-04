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
  return String(v);
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

const PAGES = [
  ["/",        "Control", "what is happening right now"],
  ["/runs",    "Runs",    "how did that run go"],
  ["/plan",    "Plan",    "what we are doing next, and why"],
  ["/train",   "Train",   "what it is being taught"],
  ["/dials",   "Dials",   "every number that gets varied, and its range"],
  ["/robot",   "Robot",   "is the physical model right"],
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
