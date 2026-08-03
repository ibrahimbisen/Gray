/* Shared by the summary and the three stage pages.

   One job: build the section list on the left, and show exactly one panel at a
   time. The page never scrolls; only the open panel does. That is the whole
   point - a section of the plan should be something you read, not something you
   scroll through looking for where it ends.

   The open section goes in the URL hash, so a section can be linked to and
   survives a refresh. */

const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function topbar(active, right) {
  const link = (href, label) =>
    `<a href="${href}" class="${href === active ? "on" : ""}">${label}</a>`;
  return `
    <div class="brand">Gray</div>
    ${link("/", "Overview")}
    ${link("/runs", "Runs")}
    ${link("/summary", "Summary")}
    ${link("/stage1", "1 &middot; Prepare")}
    ${link("/stage2", "2 &middot; Train")}
    ${link("/stage3", "3 &middot; Deploy")}
    ${link("/pose", "Pose editor")}
    <div class="sp"></div>
    <div class="state">${right || ""}</div>`;
}

/* sections: [{id, label, group, count, badge, html}]  - html may be a function */
function shell(sections, opts = {}) {
  const side = document.querySelector(".side");
  const main = document.querySelector(".main");

  let out = "";
  let group = null;
  for (const s of sections) {
    if (s.group && s.group !== group) { group = s.group; out += `<h4>${esc(group)}</h4>`; }
    out += `<button data-id="${esc(s.id)}">${
      s.badge ? `<span class="id">${esc(s.badge)}</span>` : ""
    }${esc(s.label)}${
      s.count != null ? `<span class="n">${s.count}</span>` : ""
    }</button>`;
  }
  side.innerHTML = out;

  main.innerHTML = sections.map(s =>
    `<div class="panelwrap hidden" data-panel="${esc(s.id)}">${
      typeof s.html === "function" ? s.html() : s.html}</div>`).join("");

  function open(id) {
    const known = sections.some(s => s.id === id);
    if (!known) id = sections[0].id;
    side.querySelectorAll("button").forEach(b =>
      b.classList.toggle("on", b.dataset.id === id));
    main.querySelectorAll("[data-panel]").forEach(p =>
      p.classList.toggle("hidden", p.dataset.panel !== id));
    main.scrollTop = 0;
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
    const btn = side.querySelector(`button[data-id="${CSS.escape(id)}"]`);
    if (btn) btn.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  side.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) open(b.dataset.id);
  });
  window.addEventListener("hashchange", () => open(location.hash.slice(1)));

  // Left and right arrows walk the sections, so the plan can be read straight
  // through without going back to the list every time.
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input,textarea") || e.metaKey || e.ctrlKey) return;
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    const i = sections.findIndex(s => s.id === (location.hash.slice(1) || sections[0].id));
    const next = sections[Math.min(sections.length - 1, Math.max(0, i + step))];
    if (next) open(next.id);
  });

  open(location.hash.slice(1));
  if (opts.onReady) opts.onReady(open);
}

/* One continuous page, laid out like a paper: numbered sections, the sidebar
   demoted to a contents list. Used where the point is to read end to end and
   hold two tables against each other - which shell(), showing one panel at a
   time, actively prevents.

   sections: [{id, label, count, lead, html}]  - lead:true is the unnumbered
   opening block. */
function paper(sections) {
  const side = document.querySelector(".side");
  const main = document.querySelector(".main");

  let n = 0;
  const secs = sections.map(s => ({ ...s, num: s.lead ? null : ++n }));

  side.innerHTML = `<h4>Contents</h4>` + secs.map(s =>
    `<button data-id="${esc(s.id)}">${
      s.num ? `<span class="id">${s.num}</span>` : ""
    }${esc(s.label)}${
      s.count != null ? `<span class="n">${s.count}</span>` : ""
    }</button>`).join("");

  main.innerHTML = `<div class="paper">` + secs.map(s => `
    <section id="${esc(s.id)}"${s.lead ? ` class="lead"` : ""}>
      ${s.num ? `<h2><span class="secnum">${s.num}</span>${esc(s.label)}</h2>` : ""}
      ${typeof s.html === "function" ? s.html() : s.html}
    </section>`).join("") + `</div>`;

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

  // The contents list follows the reader rather than the reader having to
  // remember where they got to. rootMargin pins "current" to the top band, so
  // a tall section stays selected the whole way down it.
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
  }, { root: main, rootMargin: "0px 0px -72% 0px" });
  main.querySelectorAll("section[id]").forEach(el => io.observe(el));

  const start = location.hash.slice(1);
  if (buttons.has(start)) go(start);
  mark(buttons.has(start) ? start : secs[0].id);
}

/* ---- small builders the pages share ---- */

const facts = (rows) => `<div class="facts">${rows.map(f => `
  <div class="fact"><div class="k">${esc(f.k)}</div>
    <div class="v">${esc(f.v)}</div>
    <div class="n">${esc(f.n ?? f.note ?? "")}</div></div>`).join("")}</div>`;

function modelBlock(m) {
  if (!m || !m.exists) return `<p class="muted">No model yet.</p>`;
  if (m.error) return `<p class="muted">Could not check: <code>${esc(m.error)}</code></p>`;
  const pct = m.total ? Math.round(100 * m.passed / m.total) : 0;
  const col = m.passed === m.total ? "var(--ok)"
            : m.passed > m.total / 2 ? "var(--warn)" : "var(--bad)";
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

function fail(err) {
  document.querySelector(".main").innerHTML =
    `<div class="panelwrap"><h2>Could not load</h2>
     <p class="muted">${esc(err)}</p></div>`;
}
