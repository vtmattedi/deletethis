// Display and configure. No DSP, no classification: every number
// here was decided by the backend.

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
