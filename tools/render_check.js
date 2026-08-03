/* Render every panel of the training centre against the REAL API payloads.
 *
 *     python run.py            # in one terminal, so there is an API to read
 *     node tools/render_check.js
 *
 * Exits non-zero if any panel throws.
 *
 * A JS error in a panel makes the page blank with nothing in the terminal to
 * say why. This runs the page's own functions outside a browser, with the same
 * JSON the server is serving right now, and fails loudly instead.
 */
const fs = require("fs");
const path = require("path");
const http = require("http");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`${url}: ${e.message}`)); }
      });
    }).on("error", reject);
  });
}

// Minimal DOM: enough for the panel builders, which only produce strings.
function stubEl(id) {
  return {
    id, value: "", checked: false, type: "text", dataset: {},
    innerHTML: "", textContent: "", style: {}, classList: { toggle(){}, add(){}, remove(){} },
    onclick: null, onchange: null,
    focus(){}, setSelectionRange(){}, addEventListener(){},
    querySelector: () => null, querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, width: 480, height: 132 }),
    setAttribute(){}, getAttribute(){},
  };
}

const document = {
  querySelector: (s) => stubEl(s),
  querySelectorAll: () => [],
  getElementById: (id) => stubEl(id),
  activeElement: null,
  addEventListener(){},
};

async function main() {
  const html = fs.readFileSync(path.join(ROOT, "dashboard/monitor.html"), "utf8");
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  if (!m) throw new Error("no <script> block found");
  // Drop the boot lines so nothing polls.
  const src = m[1].replace(/\nload\(\);\s*\nsetInterval\(load, \d+\);\s*$/, "\n");

  const monitor = await get("http://127.0.0.1:8000/api/monitor");
  const runId = (monitor.runs[0] || {}).id;
  const detail = await get(`http://127.0.0.1:8000/api/run/${encodeURIComponent(runId)}`);
  const ids = monitor.runs.slice(0, 3).map((r) => r.id);
  const compare = await get(
    `http://127.0.0.1:8000/api/compare?runs=${encodeURIComponent(ids.join(","))}&metric=reward`);

  const sandbox = {
    document, console,
    window: { addEventListener(){} },
    location: { hash: "#live" },
    history: { replaceState(){} },
    fetch: () => Promise.resolve({ ok: true, json: async () => ({}) }),
    setInterval: () => 0, setTimeout: () => 0,
    alert: () => {}, confirm: () => true,
    CSS: { escape: (s) => s },
    encodeURIComponent,
  };
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox);

  // Feed it the real state.
  vm.runInContext("STATE = __monitor; DETAIL = __detail; COMPARE_DATA = __compare;",
    Object.assign(sandbox, { __monitor: monitor, __detail: detail, __compare: compare }));
  vm.runInContext(`SELECTED = ${JSON.stringify(runId)};`, sandbox);
  vm.runInContext(`COMPARE = ${JSON.stringify(ids)}; METRIC = "reward";`, sandbox);
  vm.runInContext(`JOBLOG = "j0006"; JOBLOG_TEXT = "some log text";`, sandbox);

  const panels = ["livePanel", "queuePanel", "historyPanel", "comparePanel",
                  "jobsPanel", "planPanel"];
  let bad = 0;
  console.log(`data: ${monitor.runs.length} runs, ${monitor.queue.jobs.length} jobs, ` +
              `${detail.metrics.length} metric rows, ${compare.series.length} compare series\n`);
  for (const name of panels) {
    for (const table of [false, true]) {
      vm.runInContext(`TABLE = ${table};`, sandbox);
      try {
        const out = vm.runInContext(`${name}()`, sandbox);
        if (typeof out !== "string") throw new Error(`returned ${typeof out}`);
        if (/undefined|\[object Object\]|NaN/.test(out)) {
          const hit = out.match(/.{0,60}(undefined|\[object Object\]|NaN).{0,40}/);
          console.log(`  WARN ${name}${table ? " (table)" : ""}: ${hit[0].replace(/\s+/g, " ")}`);
        }
        if (!table) console.log(`  ok   ${name.padEnd(14)} ${out.length} chars`);
      } catch (e) {
        bad++;
        console.log(`  FAIL ${name}${table ? " (table)" : ""}: ${e.message}`);
      }
    }
  }

  // The two helpers most likely to throw on odd data.
  console.log("\nedge cases:");
  const cases = [
    ['fmt(null)', 'fmt(null)'], ['fmt("abc")', 'fmt("abc")'], ['fmt(NaN)', 'fmt(NaN)'],
    ['fmt(Infinity)', 'fmt(Infinity)'], ['fmt(0.123)', 'fmt(0.123)'], ['fmt(15000)', 'fmt(15000)'],
    ['bodySignature()', 'bodySignature().slice(0,40)'],
  ];
  for (const [label, expr] of cases) {
    try { console.log(`  ok   ${label.padEnd(18)} -> ${JSON.stringify(vm.runInContext(expr, sandbox))}`); }
    catch (e) { bad++; console.log(`  FAIL ${label}: ${e.message}`); }
  }

  // An empty-everything state must not throw either.
  console.log("\nempty state:");
  vm.runInContext(`STATE = {runs:[], active:null, metrics_available:[],
    phases:__monitor.phases, stages:__monitor.stages, next_up:__monitor.next_up,
    model:__monitor.model,
    queue:{jobs:[],queued:[],history:[],running:null,paused:false,counts:{},
           tasks:["Gray-Walk"],defaults:{},runner:{alive:false,hint:"x"}}};
    DETAIL = null; SELECTED = null; COMPARE = []; COMPARE_DATA = null; JOBLOG = null;`, sandbox);
  for (const name of panels) {
    try {
      const out = vm.runInContext(`${name}()`, sandbox);
      console.log(`  ok   ${name.padEnd(14)} ${String(out).length} chars`);
    } catch (e) { bad++; console.log(`  FAIL ${name}: ${e.message}`); }
  }

  console.log(bad ? `\n${bad} FAILURES` : "\nevery panel renders");
  process.exit(bad ? 1 : 0);
}

main().catch((e) => { console.error("harness error:", e.message); process.exit(2); });
