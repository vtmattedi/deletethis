// Display and configure. No DSP, no classification: every number
// here was decided by the backend.

import { lineChart, legend } from "./charts.js";

const FEATURES = ["rms", "30-80", "500-1k", "1k-2k", "200-1200"];

const STREAM_FIELDS = [
  ["connected", "TCP"],
  ["sampleRate", "sample rate"],
  ["lostFrames", "lost frames"],
  ["deviceDropped", "device dropped"],
  ["resyncs", "resyncs"],
  ["gaps", "gaps"],
  ["uptimeSeconds", "uptime s"],
];

const SETTINGS = [
  ["compressorThreshold", "compressor dB", "number", 0.5],
  ["fanMidThreshold", "fan 500-1k dB", "number", 0.5],
  ["fanHighThreshold", "fan 1k-2k dB", "number", 0.5],
  ["fanRequire", "fan requires", "select", ["either", "both"]],
  ["medianSeconds", "median s", "number", 0.1],
  ["holdSeconds", "hold s", "number", 0.5],
  ["eventPreSeconds", "event pre s", "number", 1],
  ["eventPostSeconds", "event post s", "number", 1],
];

const $ = (id) => document.getElementById(id);

let editing = null;   // don't fight the user while they type
let lastEventCount = -1;

function rows(table, pairs) {
  const body = table.tBodies[0];
  body.textContent = "";

  for (const [name, value] of pairs) {
    const row = body.insertRow();
    const left = row.insertCell();
    left.textContent = name;
    left.className = "name";
    row.insertCell().textContent = value;
  }
}

function buildSettings(config) {
  const form = $("settings");
  if (form.dataset.built) return;
  form.dataset.built = "1";

  for (const [key, label, kind, arg] of SETTINGS) {
    const wrap = document.createElement("label");
    const text = document.createElement("span");
    text.textContent = label;
    wrap.append(text);

    let input;
    if (kind === "select") {
      input = document.createElement("select");
      for (const option of arg) {
        const element = document.createElement("option");
        element.value = element.textContent = option;
        input.append(element);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = arg;
    }

    input.id = "cfg-" + key;
    input.value = config[key];
    input.addEventListener("focus", () => { editing = key; });
    input.addEventListener("blur", () => {
      if (editing === key) editing = null;
    });
    input.addEventListener("change", () => send(key, input));

    wrap.append(input);
    form.append(wrap);
  }
}

async function send(key, input) {
  const value = input.tagName === "SELECT"
    ? input.value
    : Number(input.value);

  const response = await fetch("/api/config", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ [key]: value }),
  });

  const note = $("saved");

  if (!response.ok) {
    note.textContent = `${key} rejected`;
    return;
  }

  const config = await response.json();
  for (const [name] of SETTINGS) {
    const field = $("cfg-" + name);
    if (field && name !== editing) field.value = config[name];
  }

  note.textContent =
    `${key} = ${config[key]} — candidate history cleared, ` +
    `the hold restarts`;
}

function applyConfig(config) {
  buildSettings(config);

  for (const [name] of SETTINGS) {
    const field = $("cfg-" + name);
    if (field && name !== editing) field.value = config[name];
  }
}

function render(data) {
  const state = data.state || "--";
  const element = $("state");
  element.textContent = state;
  element.className = "state-name state-" + state;

  $("candidate").textContent = data.candidate || "--";
  $("stable").textContent =
    data.stableSeconds == null ? "--" : data.stableSeconds.toFixed(1) + " s";

  const features = data.features || {};
  rows($("features"), FEATURES.map((name) => [
    name,
    features[name] == null ? "--" : features[name].toFixed(1) + " dB",
  ]));

  const stream = data.stream || {};
  rows($("stream"), STREAM_FIELDS.map(([key, label]) => [
    label,
    stream[key] === true ? "yes"
      : stream[key] === false ? "no"
      : stream[key] ?? "--",
  ]));

  const link = $("link");
  if (stream.connected) {
    link.textContent = "streaming";
    link.className = "pill up";
  } else {
    link.textContent = stream.lastError || "disconnected";
    link.className = "pill";
  }

  if (data.config) applyConfig(data.config);

  const written = (data.events || {}).written;
  if (written != null && written !== lastEventCount) {
    lastEventCount = written;
    loadEvents();
  }
}

async function loadEvents() {
  const response = await fetch("/api/events");
  if (!response.ok) return;

  const { events } = await response.json();
  const list = $("events");
  list.textContent = "";

  if (!events.length) {
    const item = document.createElement("li");
    item.className = "note";
    item.textContent = "No transitions recorded yet.";
    list.append(item);
    return;
  }

  for (const event of events) {
    const item = document.createElement("li");
    item.className = "event";
    item.addEventListener("click", (e) => {
      if (e.target.tagName === "AUDIO") return;   // let controls work
      openEvent(event.id, item);
    });

    const when = document.createElement("span");
    when.className = "when";
    when.textContent = event.time.replace("T", " ");

    const what = document.createElement("strong");
    what.textContent = `${event.from || "--"} → ${event.to}`;

    const held = document.createElement("span");
    held.className = "note";
    held.textContent =
      `held ${event.candidateHeldSeconds}s · ` +
      `${event.audio.seconds}s audio`;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/events/${event.id}/audio`;

    item.append(when, what, held, audio);
    list.append(item);
  }
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/live`);

  socket.onmessage = (message) => render(JSON.parse(message.data));

  socket.onclose = () => {
    $("link").textContent = "backend offline";
    $("link").className = "pill";
    setTimeout(connect, 2000);
  };
}

connect();
loadEvents();


// ------------------------------------------------------------ history

const BANDS = ["30-80", "500-1k", "1k-2k"];

async function loadHistory() {
  const seconds = Number($("range").value);

  const response = await fetch(`/api/history?seconds=${seconds}`);
  if (!response.ok) return;

  const data = await response.json();
  const c = data.columns;

  const note = $("historyNote");
  note.textContent = data.count
    ? `${data.count} point${data.count === 1 ? "" : "s"}` +
      (data.truncated ? " (truncated)" : "")
    : "nothing recorded in this range yet";

  const column = { "30-80": "band_30_80", "500-1k": "band_500_1k",
                   "1k-2k": "band_1k_2k" };

  legend($("historyLegend"), ["rms", ...BANDS]);

  lineChart(
    $("historyChart"),
    c.t || [],
    [
      { name: "rms", values: c.rms || [] },
      ...BANDS.map((b) => ({ name: b, values: c[column[b]] || [] })),
    ],
    {
      height: 200,
      states: c.state || [],
      xFormat: (v) => new Date(v * 1000).toLocaleTimeString(),
      emptyText: "no history recorded in this range yet",
    },
  );
}

$("range").addEventListener("change", loadHistory);

// ------------------------------------------------------- event detail

function table(pairs) {
  const t = document.createElement("table");
  t.className = "kv";
  for (const [k, v] of pairs) {
    const row = t.insertRow();
    row.insertCell().textContent = k;
    row.insertCell().textContent = v;
  }
  return t;
}

function heading(text) {
  const h = document.createElement("h3");
  h.textContent = text;
  return h;
}

async function openEvent(id, item) {
  for (const other of document.querySelectorAll("li.event.open")) {
    other.classList.remove("open");
  }
  if (item) item.classList.add("open");

  const panel = $("detail");
  const body = $("detailBody");
  panel.hidden = false;
  body.textContent = "loading…";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const meta = await (await fetch(`/api/events/${id}`)).json();

  body.textContent = "";

  const title = document.createElement("div");
  title.className = "state-name";
  title.style.fontSize = "24px";
  title.textContent = `${meta.from || "--"} → ${meta.to}`;
  title.classList.add("state-" + meta.to);
  body.append(title);

  const grid = document.createElement("div");
  grid.className = "grid2";

  const left = document.createElement("div");
  left.append(heading("transition"), table([
    ["time", meta.time.replace("T", " ")],
    ["candidate held", meta.candidateHeldSeconds + " s"],
    ["audio", `${meta.audio.seconds} s ` +
      `(${meta.audio.preSeconds} before / ${meta.audio.postSeconds} after)`],
  ]));

  left.append(heading("features at transition"), table(
    Object.entries(meta.featuresAtTransition).map(
      ([k, v]) => [k, v + " dB"]),
  ));

  const right = document.createElement("div");
  right.append(heading("classifier config at the time"), table(
    Object.entries(meta.classifierConfig).map(([k, v]) => [k, v]),
  ));

  const stats = document.createElement("table");
  stats.className = "kv";
  const head = stats.createTHead().insertRow();
  for (const label of ["band", "median", "min", "max", "std"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  for (const [band, s] of Object.entries(meta.decisionWindow || {})) {
    const row = stats.insertRow();
    row.insertCell().textContent = band;
    for (const key of ["median", "min", "max", "std"]) {
      row.insertCell().textContent = s[key];
    }
  }
  right.append(heading("decision window"), stats);

  grid.append(left, right);
  body.append(grid);

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = `/api/events/${id}/audio`;
  audio.style.width = "100%";
  audio.style.marginTop = "14px";
  body.append(audio);

  body.append(heading("feature timeline, recomputed from the audio"));

  const chartLegend = document.createElement("div");
  chartLegend.className = "legend";
  const chart = document.createElement("div");
  chart.className = "chart";
  const note = document.createElement("p");
  note.className = "note";
  body.append(chartLegend, chart, note);

  note.textContent = "computing…";

  const timeline = await (
    await fetch(`/api/events/${id}/timeline`)
  ).json();

  const c = timeline.columns;
  legend(chartLegend, ["rms", ...BANDS]);

  lineChart(chart, c.t, [
    { name: "rms", values: c.rms },
    ...BANDS.map((b) => ({ name: b, values: c[b] })),
  ], {
    height: 220,
    states: c.state,
    zeroLine: true,
    xFormat: (v) => v.toFixed(0) + "s",
  });

  note.textContent =
    `${timeline.count} windows, recomputed from the WAV with the ` +
    `settings recorded in the event. The strip is the published ` +
    `state; near the left edge it is still warming up, because this ` +
    `replay starts cold while the live classifier had history from ` +
    `before the pre-roll.`;
}

$("closeDetail").addEventListener("click", () => {
  $("detail").hidden = true;
  for (const other of document.querySelectorAll("li.event.open")) {
    other.classList.remove("open");
  }
});

loadHistory();
setInterval(loadHistory, 15000);
