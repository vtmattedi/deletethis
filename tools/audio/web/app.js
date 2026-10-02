// Display and configure. No DSP, no classification: every number
// here was decided by the backend.

import { lineChart, legend } from "./charts.js";

const CORE_FEATURES = ["rms", "30-80", "500-1k", "1k-2k", "200-1200"];
const ALL_FEATURES = [
  "rms", "30-80", "80-200", "200-500", "500-1k", "1k-2k",
  "2k-4k", "200-1200", "1500-4000", "peak_hz",
  "spectral_centroid", "spectral_flatness", "spectral_crest",
  "spectral_flux", "500-1k_minus_rms", "1k-2k_minus_rms",
  "2k-4k_minus_rms", "500-1k_minus_200-500",
  "1k-2k_minus_500-1k", "2k-4k_minus_500-1k",
  "1500-4000_minus_200-1200", "rms_std", "500-1k_std",
  "1k-2k_std", "2k-4k_std", "spectral_flux_median",
  "spectral_flux_std",
];

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

// Classifier v2 only. The backend rejects these on a v1 run, so they
// are shown only when it says it is running v2.
const V2_SETTINGS = [
  ["fanStabilityThreshold", "fan stability limit dB", "number", 0.25],
  ["fanStabilityFeature", "stability feature", "select", [
    "1k-2k_std", "500-1k_std", "2k-4k_std", "rms_std",
    "spectral_flux_median", "spectral_flux_std",
  ]],
  ["fanStabilityMinSeconds", "stability history s", "number", 0.25],
  ["beepMinContrastDb", "beep contrast dB", "number", 0.5],
  ["beepEdgeContrastDb", "beep edge contrast dB", "number", 0.5],
  ["beepMinLevelDb", "beep min level dB", "number", 1],
  ["beepMinMs", "beep min ms", "number", 5],
  ["beepMaxMs", "beep max ms", "number", 10],
  ["beepMinHz", "beep min Hz", "number", 5],
  ["beepMaxHz", "beep max Hz", "number", 5],
  ["beepMaxPeakSpreadHz", "beep pitch spread Hz", "number", 5],
];
let activeSettings = SETTINGS;

const $ = (id) => document.getElementById(id);
const DEFAULT_PLAYBACK_GAIN_DB = 6;

let lastEventCount = -1;
let settingsDirty = false;
let eventPage = 1;
let eventPages = 0;
const EVENT_PAGE_SIZE = 10;
const selectedEventIds = new Set();
// Every event the list has shown, so a bulk review knows which schema
// each selected one takes.
const knownEvents = new Map();
let visibleEventIds = [];
let rangeSelectionAnchor = null;
let audioPlaybackContext = null;
const boostedAudio = new WeakMap();
const activePlaybackGains = new Set();
let playbackGainDb = DEFAULT_PLAYBACK_GAIN_DB;
let showAllFeatures = false;
let lastFeatures = {};
let liveAudioSocket = null;
let liveAudioGain = null;
let liveAudioSampleRate = 0;
let nextLiveAudioTime = 0;
const liveAudioSources = new Set();

try {
  showAllFeatures = localStorage.getItem("showAllFeatures") === "true";
  const savedGainText = localStorage.getItem("playbackGainDb");
  const savedGain = Number(savedGainText);
  if (savedGainText !== null && Number.isFinite(savedGain)) {
    playbackGainDb = Math.max(0, Math.min(18, savedGain));
  }
} catch (_) {
  // Storage can be disabled; the toggle still works for this page load.
}

function enablePlaybackGain(audio) {
  audio.title = `Playback gain +${playbackGainDb} dB; WAV unchanged`;
  audio.addEventListener("play", async () => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;

    if (!audioPlaybackContext) audioPlaybackContext = new AudioContext();
    if (!boostedAudio.has(audio)) {
      const source = audioPlaybackContext.createMediaElementSource(audio);
      const gain = audioPlaybackContext.createGain();
      gain.gain.value = 10 ** (playbackGainDb / 20);
      source.connect(gain).connect(audioPlaybackContext.destination);
      boostedAudio.set(audio, { source, gain });
      activePlaybackGains.add(gain);
    }
    if (audioPlaybackContext.state === "suspended") {
      await audioPlaybackContext.resume();
    }
  });
}

function releasePlaybackAudio(container) {
  for (const audio of container.querySelectorAll("audio")) {
    const nodes = boostedAudio.get(audio);
    if (nodes) {
      nodes.source.disconnect();
      nodes.gain.disconnect();
      activePlaybackGains.delete(nodes.gain);
      boostedAudio.delete(audio);
    }
  }
}

function setPlaybackGain(value) {
  const numeric = Number(value);
  playbackGainDb = Number.isFinite(numeric)
    ? Math.max(0, Math.min(18, numeric))
    : DEFAULT_PLAYBACK_GAIN_DB;
  $("playbackGain").value = playbackGainDb;
  $("playbackGainValue").textContent = `+${playbackGainDb} dB`;
  for (const gain of activePlaybackGains) {
    gain.gain.value = 10 ** (playbackGainDb / 20);
  }
  for (const audio of document.querySelectorAll("audio")) {
    audio.title = `Playback gain +${playbackGainDb} dB; WAV unchanged`;
  }
  try {
    localStorage.setItem("playbackGainDb", String(playbackGainDb));
  } catch (_) {
    // Keep the preference for this page when storage is disabled.
  }
}

$("playbackGain").addEventListener("input", (event) => {
  setPlaybackGain(event.target.value);
});
setPlaybackGain(playbackGainDb);

function stopLiveAudio(message = "Live listening stopped.") {
  const socket = liveAudioSocket;
  liveAudioSocket = null;
  if (socket && socket.readyState < WebSocket.CLOSING) {
    try { socket.close(); } catch (_) { /* Already closing. */ }
  }

  for (const source of liveAudioSources) {
    source.onended = null;
    try { source.stop(); } catch (_) { /* Source already ended. */ }
    source.disconnect();
  }
  liveAudioSources.clear();
  if (liveAudioGain) {
    activePlaybackGains.delete(liveAudioGain);
    liveAudioGain.disconnect();
    liveAudioGain = null;
  }
  liveAudioSampleRate = 0;
  nextLiveAudioTime = 0;

  const button = $("toggleLiveAudio");
  button.textContent = "Listen live";
  button.classList.remove("listening");
  button.setAttribute("aria-pressed", "false");
  $("liveAudioNote").textContent = message;
}

function scheduleLiveAudio(packet) {
  if (!audioPlaybackContext || !liveAudioGain || !liveAudioSampleRate) return;

  const pcm = new Int32Array(packet);
  const buffer = audioPlaybackContext.createBuffer(
    1, pcm.length, liveAudioSampleRate,
  );
  const channel = buffer.getChannelData(0);
  for (let index = 0; index < pcm.length; index += 1) {
    channel[index] = Math.max(-1, Math.min(1, pcm[index] / 8388608));
  }

  const now = audioPlaybackContext.currentTime;
  if (nextLiveAudioTime < now + 0.04 || nextLiveAudioTime > now + 0.5) {
    nextLiveAudioTime = now + 0.08;
  }
  const source = audioPlaybackContext.createBufferSource();
  source.buffer = buffer;
  source.connect(liveAudioGain);
  source.onended = () => {
    source.disconnect();
    liveAudioSources.delete(source);
  };
  liveAudioSources.add(source);
  source.start(nextLiveAudioTime);
  nextLiveAudioTime += buffer.duration;
}

async function startLiveAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) {
    $("liveAudioNote").textContent = "Live audio is unsupported in this browser.";
    return;
  }

  if (!audioPlaybackContext) audioPlaybackContext = new AudioContext();
  await audioPlaybackContext.resume();
  liveAudioGain = audioPlaybackContext.createGain();
  liveAudioGain.gain.value = 10 ** (playbackGainDb / 20);
  liveAudioGain.connect(audioPlaybackContext.destination);
  activePlaybackGains.add(liveAudioGain);

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/audio`);
  socket.binaryType = "arraybuffer";
  liveAudioSocket = socket;

  const button = $("toggleLiveAudio");
  button.textContent = "Stop listening";
  button.classList.add("listening");
  button.setAttribute("aria-pressed", "true");
  $("liveAudioNote").textContent = "Connecting live audio…";

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "format") {
        liveAudioSampleRate = message.sampleRate;
        nextLiveAudioTime = 0;
        $("liveAudioNote").textContent =
          `Listening live at ${message.sampleRate} Hz · headphones recommended.`;
      } else if (message.type === "status" && !message.connected) {
        $("liveAudioNote").textContent = "Waiting for the audio stream…";
      }
      return;
    }
    scheduleLiveAudio(event.data);
  };
  socket.onclose = () => {
    if (liveAudioSocket === socket) {
      stopLiveAudio("Live audio disconnected.");
    }
  };
  socket.onerror = () => {
    $("liveAudioNote").textContent = "Live audio connection failed.";
  };
}

$("toggleLiveAudio").addEventListener("click", () => {
  if (liveAudioSocket) stopLiveAudio();
  else startLiveAudio();
});

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

function formatFeatureValue(name, value) {
  if (value == null) return "--";
  if (name === "peak_hz" || name === "spectral_centroid") {
    return value.toFixed(0) + " Hz";
  }
  if (name.startsWith("spectral_")) return value.toFixed(4);
  return value.toFixed(2) + " dB";
}

function updateFeatureToggle() {
  const button = $("toggleFeatures");
  button.textContent = showAllFeatures ? "Show core" : "Show all";
  button.setAttribute("aria-expanded", String(showAllFeatures));
}

function renderFeatures(features) {
  lastFeatures = features;
  const names = showAllFeatures ? ALL_FEATURES : CORE_FEATURES;
  rows($("features"), names.map((name) => [
    name,
    formatFeatureValue(name, features[name]),
  ]));
}

$("toggleFeatures").addEventListener("click", () => {
  showAllFeatures = !showAllFeatures;
  try {
    localStorage.setItem("showAllFeatures", String(showAllFeatures));
  } catch (_) {
    // Keep the in-memory preference when browser storage is disabled.
  }
  updateFeatureToggle();
  renderFeatures(lastFeatures);
});

updateFeatureToggle();

function buildSettings(config) {
  const form = $("settings");
  if (form.dataset.built) return;
  form.dataset.built = "1";

  activeSettings = config.classifierVersion === "v2"
    ? [...SETTINGS, ...V2_SETTINGS]
    : SETTINGS;

  for (const [key, label, kind, arg] of activeSettings) {
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
    input.addEventListener("input", () => {
      settingsDirty = true;
      $("saveSettings").disabled = false;
      $("saved").textContent = "Unsaved changes";
    });

    wrap.append(input);
    form.append(wrap);
  }
}

async function saveSettings() {
  const changes = {};
  for (const [key, , kind] of activeSettings) {
    const input = $("cfg-" + key);
    if (!input.value) {
      $("saved").textContent = "Fill in every setting before saving";
      return;
    }
    changes[key] = kind === "select" ? input.value : Number(input.value);
  }

  const response = await fetch("/api/config", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });

  const note = $("saved");

  if (!response.ok) {
    note.textContent = "Settings rejected — check the entered values";
    return;
  }

  const config = await response.json();
  for (const [name] of activeSettings) {
    const field = $("cfg-" + name);
    if (field) field.value = config[name];
  }

  settingsDirty = false;
  $("saveSettings").disabled = true;
  note.textContent = "Saved — candidate history cleared; the hold restarted";
}

function applyConfig(config) {
  buildSettings(config);

  if (!settingsDirty) {
    for (const [name] of activeSettings) {
      const field = $("cfg-" + name);
      if (field) field.value = config[name];
    }
  }
}

$("saveSettings").addEventListener("click", saveSettings);

// The TCP address the backend reads audio from. Separate from the
// classifier settings: it is not part of /api/config and applying it
// reconnects rather than retuning.
let targetDirty = false;

function applyTarget(target) {
  if (targetDirty || !target) return;
  const input = $("targetInput");
  if (document.activeElement !== input) input.value = target;
}

$("targetInput").addEventListener("input", () => {
  targetDirty = true;
  $("applyTarget").disabled = !$("targetInput").value.trim();
  $("targetNote").textContent = "Unsaved address";
});

async function saveTarget() {
  const note = $("targetNote");
  const target = $("targetInput").value.trim();
  if (!target) return;

  $("applyTarget").disabled = true;

  let response;
  try {
    response = await fetch("/api/stream/target", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
  } catch (_) {
    note.textContent = "Could not reach the backend";
    $("applyTarget").disabled = false;
    return;
  }

  if (!response.ok) {
    // 422 carries pydantic's message for the field.
    let detail = "";
    try {
      const body = await response.json();
      detail = Array.isArray(body.detail)
        ? body.detail.map((item) => item.msg.replace(/^Value error, /, "")).join("; ")
        : String(body.detail || "");
    } catch (_) {}
    note.textContent = "Address rejected" + (detail ? ": " + detail : "");
    $("applyTarget").disabled = false;
    return;
  }

  const saved = await response.json();
  targetDirty = false;
  $("targetInput").value = saved.target;
  note.textContent = "Saved — reconnecting to " + saved.target;
}

$("applyTarget").addEventListener("click", saveTarget);
$("targetForm").addEventListener("submit", (event) => {
  event.preventDefault();
  saveTarget();
});

const settingsDialog = $("settingsDialog");
$("openSettings").addEventListener("click", () => settingsDialog.showModal());
$("closeSettings").addEventListener("click", () => settingsDialog.close());
// A click on the backdrop (the dialog element itself) closes it.
settingsDialog.addEventListener("click", (event) => {
  if (event.target === settingsDialog) settingsDialog.close();
});

function setObservation(id, value) {
  const element = $(id);
  if (value === true) {
    element.textContent = "DETECTED";
    element.className = "obs-value yes";
  } else if (value === false) {
    element.textContent = "not detected";
    element.className = "obs-value no";
  } else {
    element.textContent = "waiting…";
    element.className = "obs-value unknown";
  }
}

function commandAge(epoch) {
  const age = Math.max(0, Date.now() / 1000 - epoch);
  return age < 90 ? `${age.toFixed(0)} s ago`
    : age < 5400 ? `${(age / 60).toFixed(0)} min ago`
    : new Date(epoch * 1000).toLocaleTimeString();
}

const yesNo = (value) => value === true ? "yes" : value === false ? "no" : "--";
const seconds = (value) => value == null ? "--" : `${value.toFixed(1)} s`;

function renderClassifier(data) {
  const config = data.config || {};
  const version = data.classifierVersion || config.classifierVersion;
  const v2 = version === "v2";

  $("classifierVersion").textContent = version ? version.toUpperCase() : "--";

  // v1 shows its one state; v2 shows three separate observations.
  $("stateCard").hidden = v2;
  for (const id of ["fanCard", "compressorCard", "beepCard", "v2Note",
                    "commandCard"]) {
    $(id).hidden = !v2;
  }
  if (!v2) return;

  const seen = data.observations || {};
  const candidates = data.candidates || {};
  const held = data.stableSeconds || {};
  const evidence = data.fanEvidence || {};
  const features = data.features || {};

  const last = data.lastCommand;
  $("lastCommand").textContent = last
    ? `Last command: ${last.command} · ${commandAge(last.time)}` +
      `${last.note ? ` · ${last.note}` : ""}`
    : "No command marked yet.";

  setObservation("fanValue", seen.fan);
  $("fanCandidate").textContent = yesNo(candidates.fan);
  $("fanStable").textContent = seconds(held.fan);
  $("fanEnergy").textContent = evidence.energy == null
    ? "--"
    : evidence.energy ? "present" : "absent";

  const feature = config.fanStabilityFeature || "1k-2k_std";
  const value = features[feature];
  const limit = config.fanStabilityThreshold;
  $("fanStability").textContent = value == null
    ? "--"
    : `${formatFeatureValue(feature, value)} / ${limit} dB`;
  // Steady means at or under the limit: both tests must pass for a fan.
  $("fanStability").className =
    value == null || limit == null ? "" : value <= limit ? "steady" : "moving";

  setObservation("compressorValue", seen.compressor);
  $("compressorCandidate").textContent = yesNo(candidates.compressor);
  $("compressorStable").textContent = seconds(held.compressor);
  const level = features["30-80"];
  $("compressorLevel").textContent = level == null
    ? "--"
    : `${level.toFixed(1)} dB (limit ${config.compressorThreshold})`;

  const beep = data.lastBeep;
  const count = data.beepCount ?? 0;
  const beepValue = $("beepValue");
  $("beepCount").textContent = String(count);
  if (beep) {
    const ago = Math.max(0, (data.streamSeconds ?? 0) - beep.streamSeconds);
    beepValue.textContent = ago < 2 ? "BEEP" : `${ago.toFixed(0)} s ago`;
    beepValue.className = ago < 2 ? "obs-value yes" : "obs-value no";
    $("beepDuration").textContent = `${beep.durationMs.toFixed(0)} ms`;
    $("beepPitch").textContent = `${beep.peakHz.toFixed(0)} Hz`;
    $("beepContrast").textContent = `${beep.contrastDb.toFixed(1)} dB`;
  } else {
    beepValue.textContent = "none yet";
    beepValue.className = "obs-value unknown";
    for (const id of ["beepDuration", "beepPitch", "beepContrast"]) {
      $(id).textContent = "--";
    }
  }
}

function render(data) {
  // v1 has one combined state; v2 has none (see renderClassifier), and
  // its stableSeconds is per observation rather than a number.
  if (data.classifierVersion !== "v2") {
    const state = data.state || "--";
    const element = $("state");
    element.textContent = state;
    element.className = "state-name state-" + state;

    $("candidate").textContent = data.candidate || "--";
    $("stable").textContent = typeof data.stableSeconds === "number"
      ? data.stableSeconds.toFixed(1) + " s" : "--";
  }

  renderFeatures(data.features || {});
  renderClassifier(data);

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
  applyTarget(data.target);

  const eventStatus = data.events || {};
  const pending = eventStatus.pending || 0;
  $("eventCaptureStatus").textContent = pending
    ? `${pending} capture${pending === 1 ? "" : "s"} finishing post-roll…`
    : "";

  const written = eventStatus.written;
  if (written != null && written !== lastEventCount) {
    lastEventCount = written;
    loadEvents();
  }
}

async function loadEvents() {
  const params = new URLSearchParams({
    page: eventPage,
    pageSize: EVENT_PAGE_SIZE,
    search: $("eventSearch").value.trim(),
  });
  const response = await fetch(`/api/events?${params}`);
  if (!response.ok) return;

  const data = await response.json();
  const { events } = data;
  if (data.pages && eventPage > data.pages) {
    eventPage = data.pages;
    return loadEvents();
  }
  eventPages = data.pages;
  visibleEventIds = events.map((event) => event.id);
  rangeSelectionAnchor = null;
  const list = $("events");
  releasePlaybackAudio(list);
  list.textContent = "";
  $("eventCount").textContent = `${data.total} saved`;
  $("eventPage").textContent = eventPages
    ? `Page ${data.page} of ${eventPages}` : "No pages";
  $("previousEvents").disabled = data.page <= 1;
  $("nextEvents").disabled = data.page >= eventPages;
  updateEventSelection();

  if (!events.length) {
    const item = document.createElement("li");
    item.className = "note";
    item.textContent = "No saved events match.";
    list.append(item);
    return;
  }

  for (const event of events) {
    const item = document.createElement("li");
    item.className = "event";
    item.dataset.eventId = event.id;
    item.addEventListener("click", (e) => {
      if (e.target.closest("audio, button, input, label")) return;
      openEvent(event.id, item);
    });

    const select = document.createElement("input");
    select.type = "checkbox";
    select.className = "event-select";
    select.dataset.eventId = event.id;
    select.checked = selectedEventIds.has(event.id);
    select.setAttribute("aria-label", `Select event ${event.id}`);
    select.addEventListener("click", (clickEvent) => {
      const anchorIndex = visibleEventIds.indexOf(rangeSelectionAnchor);
      const currentIndex = visibleEventIds.indexOf(event.id);
      if (clickEvent.shiftKey && anchorIndex >= 0 && currentIndex >= 0) {
        const start = Math.min(anchorIndex, currentIndex);
        const end = Math.max(anchorIndex, currentIndex);
        for (const identifier of visibleEventIds.slice(start, end + 1)) {
          if (select.checked) selectedEventIds.add(identifier);
          else selectedEventIds.delete(identifier);
        }
        for (const checkbox of document.querySelectorAll(".event-select")) {
          checkbox.checked = selectedEventIds.has(checkbox.dataset.eventId);
        }
      } else if (select.checked) {
        selectedEventIds.add(event.id);
      } else {
        selectedEventIds.delete(event.id);
      }
      rangeSelectionAnchor = event.id;
      updateEventSelection();
    });

    const when = document.createElement("span");
    when.className = "when";
    when.textContent = event.time.replace("T", " ");

    knownEvents.set(event.id, event);

    const what = document.createElement("strong");
    what.textContent = eventLabel(event);
    if (isV2(event)) {
      what.className = `event-kind kind-${event.eventType}`;
    }

    const review = document.createElement("span");
    const verdict = verdictOf(event.review);
    let correctness = verdict === null
      ? "unreviewed"
      : verdict ? "correct" : "incorrect";
    review.className = `review-status review-${correctness}`;
    const outcome = event.review?.outcome;
    review.textContent = outcome === "stayed_true" ? "no transition: true"
      : outcome === "stayed_false" ? "no transition: false"
      : correctness;

    // Quick review. v1: one "Correct". v2 fan/compressor: the three
    // outcomes of a reported X -> Y; a beep: cor. Manual: open it.
    const quick = document.createElement("span");
    quick.className = "quick-reviews";
    const observation = isV2(event) &&
      (event.eventType === "fan" || event.eventType === "compressor");
    const choices = !isV2(event)
      ? [["Correct", null, "Mark correct with no interference or notes"]]
      : observation
        ? [
          ["cor", "correct", `Correct: it really went ${event.from} → ${event.to}`],
          ["ntt", "stayed_true", "No transition: it stayed true"],
          ["ntf", "stayed_false", "No transition: it stayed false"],
        ]
        : event.eventType === "beep"
          ? [["cor", "correct", "Correct: there really was a beep"]]
          : [];
    for (const [label, outcome, title] of choices) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "quick-correct";
      button.textContent = label;
      button.title = title;
      button.classList.toggle(
        "active",
        isV2(event) && event.review?.status === "reviewed" &&
          (event.review.outcome ?? (event.review.correct ? "correct" : null))
            === outcome,
      );
      button.addEventListener("click", async () => {
        for (const other of quick.querySelectorAll("button")) {
          other.disabled = true;
        }
        const body = !isV2(event)
          ? {
            classificationCorrect: true,
            actualFrom: event.from || "UNKNOWN",
            actualTo: event.to || "UNKNOWN",
            interference: [], notes: "",
          }
          : observation
            ? { outcome, interference: [], notes: "" }
            : { correct: true, interference: [], notes: "" };
        try {
          const response = await fetch(`/api/events/${event.id}/review`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (!response.ok) throw new Error("review request failed");
          const reopen = item.classList.contains("open");
          await loadEvents();
          const fresh = document.querySelector(
            `li.event[data-event-id="${event.id}"]`,
          );
          if (reopen && fresh) openEvent(event.id, fresh);
        } catch (_) {
          button.textContent = "retry";
          for (const other of quick.querySelectorAll("button")) {
            other.disabled = false;
          }
        }
      });
      quick.append(button);
    }

    const held = document.createElement("span");
    held.className = "note";
    held.textContent =
      `${event.source === "manual" ? "manual · " : ""}` +
      `${isV2(event) ? "" : `held ${event.candidateHeldSeconds}s · `}` +
      `${event.audio.seconds}s audio`;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/events/${event.id}/audio`;
    enablePlaybackGain(audio);

    item.append(select, when, what);
    if (event.commandContext) {
      const sent = document.createElement("span");
      sent.className = "event-kind kind-command";
      const offset = event.commandContext.secondsFromEvent;
      sent.textContent = `sent ${event.commandContext.command}` +
        `${offset == null ? "" : ` (${offset > 0 ? "+" : ""}${offset} s)`}`;
      sent.title = "A command was marked near this event";
      item.append(sent);
    }
    item.append(review, quick, held, audio);
    list.append(item);
  }
}

const isV2 = (event) => event.classifierVersion === "v2" && !!event.eventType;

// The review verdict, whichever schema the event uses.
function verdictOf(review) {
  if (review?.status !== "reviewed") return null;
  return isV2Review(review) ? review.correct : review.classificationCorrect;
}
const isV2Review = (review) => review && review.correct !== undefined;

function eventLabel(event) {
  if (!isV2(event)) return `${event.from || "--"} → ${event.to}`;

  switch (event.eventType) {
    case "fan": return event.to ? "FAN ON" : "FAN OFF";
    case "compressor": return event.to ? "COMPRESSOR ON" : "COMPRESSOR OFF";
    case "beep": {
      const b = event.beep || {};
      return b.peakHz
        ? `BEEP · ${b.durationMs.toFixed(0)} ms at ${b.peakHz.toFixed(0)} Hz`
        : "BEEP";
    }
    default: return event.eventType.toUpperCase();
  }
}

function updateEventSelection() {
  const count = selectedEventIds.size;
  $("selectionCount").textContent = `${count} selected`;
  $("reviewSelectedEvents").disabled = count === 0;
  $("deleteSelectedEvents").disabled = count === 0;

  const selectAll = $("selectAllEvents");
  const visibleSelected = visibleEventIds.filter(
    (identifier) => selectedEventIds.has(identifier),
  ).length;
  selectAll.checked = visibleEventIds.length > 0 &&
    visibleSelected === visibleEventIds.length;
  selectAll.indeterminate = visibleSelected > 0 &&
    visibleSelected < visibleEventIds.length;
  selectAll.disabled = visibleEventIds.length === 0;
}

function closeBulkReview() {
  $("bulkReview").hidden = true;
  $("bulkReviewBody").textContent = "";
}

function openBulkReview() {
  const identifiers = [...selectedEventIds];
  if (!identifiers.length) return;

  // The two classifier versions are reviewed differently, so one
  // request covers one version.
  const known = identifiers.map((id) => knownEvents.get(id));
  const v2Count = known.filter((event) => event && isV2(event)).length;
  const v1Count = known.filter((event) => event && !isV2(event)).length;
  if (v2Count + v1Count !== identifiers.length || (v2Count && v1Count)) {
    $("bulkActionNote").textContent =
      "Select events of one classifier version, from the list, to review " +
      "them together.";
    return;
  }
  const v2 = v2Count > 0;
  if (known.some((event) => event.eventType === "manual")) {
    $("bulkActionNote").textContent =
      "Manual captures are reviewed one at a time: open each and say what " +
      "was happening.";
    return;
  }

  const panel = $("bulkReview");
  const body = $("bulkReviewBody");
  $("bulkActionNote").textContent = "";
  $("bulkReviewSummary").textContent =
    `This review will be applied to ${identifiers.length} selected event` +
    `${identifiers.length === 1 ? "" : "s"}. ` + (v2
      ? "Correct means the detector was right about each one."
      : "Correct uses each event's own classifier transition.");
  body.textContent = "";

  const form = document.createElement("div");
  form.className = "review-form";
  let verdict = null;

  const choices = document.createElement("div");
  choices.className = "review-actions";
  const correct = document.createElement("button");
  const wrong = document.createElement("button");
  const stayedTrue = document.createElement("button");
  const stayedFalse = document.createElement("button");
  const save = document.createElement("button");
  const message = document.createElement("span");
  correct.type = wrong.type = save.type = "button";
  stayedTrue.type = stayedFalse.type = "button";
  correct.textContent = "Correct";
  wrong.textContent = v2 ? "Not a beep" : "Wrong";
  stayedTrue.textContent = "No transition: true";
  stayedFalse.textContent = "No transition: false";
  save.textContent = "Save reviews";
  correct.className = wrong.className = "review-choice";
  stayedTrue.className = stayedFalse.className = "review-choice";
  message.className = "note";
  choices.append(
    correct, ...(v2 ? [stayedTrue, stayedFalse] : []), wrong, save, message,
  );
  // v2: fan / compressor events take the No transition outcomes, beeps
  // take Correct or Not a beep. One click applies to the whole selection.
  let outcome = null;
  form.append(choices);

  const stateSelect = () => {
    const select = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Select…";
    select.append(blank);
    for (const state of ["OFF", "FAN", "COMPRESSOR", "UNKNOWN"]) {
      const option = document.createElement("option");
      option.value = option.textContent = state;
      select.append(option);
    }
    return select;
  };
  const field = (labelText, control) => {
    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = labelText;
    label.append(name, control);
    form.append(label);
  };

  // v1 says what the state really was before and after. v2 is binary:
  // Wrong means the opposite of what the event reported.
  const actualFrom = stateSelect();
  const actualTo = stateSelect();
  if (!v2) {
    field("Actual from", actualFrom);
    field("Actual to", actualTo);
  }

  const tags = document.createElement("div");
  tags.className = "review-tags";
  for (const tag of ["talking", "printer", "tv", "other"]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    const text = document.createElement("span");
    input.type = "checkbox";
    input.value = tag;
    text.textContent = tag;
    label.append(input, text);
    tags.append(label);
  }
  field("Interference", tags);

  const notes = document.createElement("textarea");
  notes.placeholder = "Optional notes applied to every selected event";
  field("Notes", notes);

  const showVerdict = () => {
    correct.classList.toggle("active", verdict === true);
    wrong.classList.toggle("active", verdict === false);
    stayedTrue.classList.toggle("active", outcome === "stayed_true");
    stayedFalse.classList.toggle("active", outcome === "stayed_false");
    actualFrom.disabled = verdict === true;
    actualTo.disabled = verdict === true;
    if (verdict === true) {
      actualFrom.value = "";
      actualTo.value = "";
    }
  };
  stayedTrue.addEventListener("click", () => {
    verdict = false;
    outcome = "stayed_true";
    showVerdict();
    message.textContent = "Beeps in the selection cannot take this.";
  });
  stayedFalse.addEventListener("click", () => {
    verdict = false;
    outcome = "stayed_false";
    showVerdict();
    message.textContent = "Beeps in the selection cannot take this.";
  });
  correct.addEventListener("click", () => {
    verdict = true;
    outcome = null;
    showVerdict();
    message.textContent = v2
      ? "Each event will be confirmed as the detector saw it."
      : "Each event will use its own classifier labels.";
  });
  wrong.addEventListener("click", () => {
    verdict = false;
    outcome = null;
    showVerdict();
    message.textContent = "";
  });
  showVerdict();

  save.addEventListener("click", async () => {
    if (verdict === null) {
      message.textContent = "Choose Correct or Wrong.";
      return;
    }
    if (verdict === false && !v2 && (!actualFrom.value || !actualTo.value)) {
      message.textContent = "Choose both actual states for an incorrect review.";
      return;
    }

    const payload = {
      ids: identifiers,
      interference: [...tags.querySelectorAll("input:checked")]
        .map((input) => input.value),
      notes: notes.value,
    };
    if (v2) {
      if (outcome) payload.outcome = outcome;
      else payload.correct = verdict;
    } else {
      payload.classificationCorrect = verdict;
      if (verdict === false) {
        payload.actualFrom = actualFrom.value;
        payload.actualTo = actualTo.value;
      }
    }

    save.disabled = true;
    message.textContent = "Saving…";
    const response = await fetch("/api/events/reviews", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      save.disabled = false;
      const detail = (await response.json().catch(() => ({}))).detail;
      message.textContent = detail?.message ||
        "Reviews could not be saved. Refresh and retry.";
      return;
    }

    const result = await response.json();
    for (const identifier of result.updated) selectedEventIds.delete(identifier);
    closeBulkReview();
    await loadEvents();
    $("bulkActionNote").textContent =
      `${result.count} review${result.count === 1 ? "" : "s"} saved.`;
  });

  body.append(form);
  panel.hidden = false;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function deleteSelectedEvents() {
  const identifiers = [...selectedEventIds];
  if (!identifiers.length) return;
  if (!confirm(
    `Delete ${identifiers.length} selected event` +
    `${identifiers.length === 1 ? "" : "s"} and their WAV files permanently?`,
  )) return;

  const button = $("deleteSelectedEvents");
  button.disabled = true;
  button.textContent = "Deleting…";
  const response = await fetch("/api/events", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: identifiers }),
  });

  if (!response.ok) {
    button.textContent = "Delete failed — retry";
    button.disabled = false;
    return;
  }

  const result = await response.json();
  for (const identifier of result.deleted) selectedEventIds.delete(identifier);
  for (const identifier of result.notFound) selectedEventIds.delete(identifier);
  button.textContent = "Delete selected";
  closeBulkReview();
  $("detail").hidden = true;
  await loadEvents();
}

$("selectAllEvents").addEventListener("change", (event) => {
  for (const identifier of visibleEventIds) {
    if (event.target.checked) selectedEventIds.add(identifier);
    else selectedEventIds.delete(identifier);
  }
  for (const checkbox of document.querySelectorAll(".event-select")) {
    checkbox.checked = event.target.checked;
  }
  updateEventSelection();
});
$("deleteSelectedEvents").addEventListener("click", deleteSelectedEvents);
$("reviewSelectedEvents").addEventListener("click", openBulkReview);
$("closeBulkReview").addEventListener("click", closeBulkReview);

async function recordEvent() {
  const button = $("recordEvent");
  const note = $("recordNote");
  button.disabled = true;
  note.textContent = "Starting…";
  const response = await fetch("/api/events/manual", { method: "POST" });
  const result = await response.json();
  button.disabled = false;
  note.textContent = response.ok
    ? `Recording ${result.id}; it will appear after the post-roll.`
    : (result.detail || "Could not start recording");
}

$("recordEvent").addEventListener("click", recordEvent);

async function markCommand() {
  const command = $("commandName").value.trim();
  const note = $("commandNote");
  if (!command) {
    note.textContent = "Name the command first.";
    return;
  }
  const response = await fetch("/api/commands", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      command,
      expectedBeep: $("commandExpectBeep").checked,
    }),
  });
  note.textContent = response.ok
    ? `Marked ${command} at ${new Date().toLocaleTimeString()}. Events ` +
      "within 10 s will carry it as context."
    : "Could not record the command.";
}

$("markCommand").addEventListener("click", markCommand);
$("eventSearchButton").addEventListener("click", () => {
  eventPage = 1;
  loadEvents();
});
$("eventSearch").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    eventPage = 1;
    loadEvents();
  }
});
$("previousEvents").addEventListener("click", () => {
  if (eventPage > 1) eventPage -= 1;
  loadEvents();
});
$("nextEvents").addEventListener("click", () => {
  if (eventPage < eventPages) eventPage += 1;
  loadEvents();
});

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
    ? `${data.count} of ${data.total} point` +
      (data.total === 1 ? "" : "s") +
      (data.downsampled
        ? ` · averaged over ${data.bucketSeconds}s buckets`
        : "")
    : "nothing recorded in this range yet";

  // Turn connect/disconnect events into [from, to] outage spans.
  const outages = [];
  let downAt = null;

  for (const event of data.connection || []) {
    if (!event.connected && downAt == null) {
      downAt = Math.max(event.t, data.from);
    } else if (event.connected && downAt != null) {
      outages.push([downAt, event.t]);
      downAt = null;
    }
  }

  if (downAt != null) outages.push([downAt, null]);

  const column = { "30-80": "band_30_80", "500-1k": "band_500_1k",
                   "1k-2k": "band_1k_2k" };

  legend($("historyLegend"), ["rms", ...BANDS]);

  // v2 rows carry the two observations (a downsampled bucket holds the
  // fraction of it that was on) and a running beep count.
  const asBool = (v) => v == null ? null : v >= 0.5;
  const hasV2 = (c.fan_detected || []).some((v) => v != null);
  const counts = c.beep_count || [];
  const beepMarks = [];
  for (let i = 1; i < counts.length; i++) {
    if (counts[i] != null && counts[i - 1] != null && counts[i] > counts[i - 1]) {
      beepMarks.push({ t: c.t[i] });
    }
  }

  const commandMarks = (data.commands || []).map(
    (cmd) => ({ t: cmd.time, colour: "#d6a8f0" }),
  );
  if (commandMarks.length) {
    legend($("historyLegend"), ["rms", ...BANDS]);
    const key = document.createElement("span");
    key.className = "key";
    key.innerHTML = '<i style="background:#d6a8f0"></i>command sent';
    $("historyLegend").append(key);
    if (hasV2) {
      const beepKey = document.createElement("span");
      beepKey.className = "key";
      beepKey.innerHTML = '<i style="background:#e0c341"></i>beep';
      $("historyLegend").append(beepKey);
    }
  }

  lineChart(
    $("historyChart"),
    c.t || [],
    [
      { name: "rms", values: c.rms || [] },
      ...BANDS.map((b) => ({ name: b, values: c[column[b]] || [] })),
    ],
    {
      height: 200,
      ...(hasV2
        ? {
          strips: [
            { name: "fan", values: c.fan_detected.map(asBool), colour: "#1f9d55" },
            { name: "comp", values: c.compressor_detected.map(asBool), colour: "#d2691e" },
          ],
          marks: [...beepMarks, ...commandMarks],
        }
        : { states: c.state || [], marks: commandMarks }),
      outages,
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

function closeEventDetail() {
  const body = $("detailBody");
  for (const audio of body.querySelectorAll("audio")) audio.pause();
  releasePlaybackAudio(body);
  $("detail").hidden = true;
  for (const item of document.querySelectorAll("li.event.open")) {
    item.classList.remove("open");
  }
}

// Manual captures claim nothing, so the review says what was happening.
function buildReviewManual(meta) {
  const form = document.createElement("div");
  form.className = "review-form";
  form.append(heading("Review"));

  const review = meta.review || {};
  const seen = meta.observations || {};
  const choice = (value) => {
    const select = document.createElement("select");
    for (const [text, key] of [["Select…", ""], ["yes", "yes"], ["no", "no"]]) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = text;
      select.append(option);
    }
    select.value = value === true ? "yes" : value === false ? "no" : "";
    return select;
  };
  const field = (label, control) => {
    const wrapper = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    wrapper.append(name, control);
    form.append(wrapper);
  };

  // Start from what was already said, else from what the detectors saw
  // (beeps have no such guess).
  const fan = choice(review.actualFan ?? seen.fan);
  const compressor = choice(review.actualCompressor ?? seen.compressor);
  const beep = choice(review.actualBeep);
  field("Fan running", fan);
  field("Compressor running", compressor);
  field("Had a beep", beep);

  const tags = document.createElement("div");
  tags.className = "review-tags";
  const selected = new Set(review.interference || []);
  for (const tag of ["talking", "printer", "tv", "other"]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = tag;
    input.checked = selected.has(tag);
    const text = document.createElement("span");
    text.textContent = tag;
    label.append(input, text);
    tags.append(label);
  }
  field("Interference", tags);

  const notes = document.createElement("textarea");
  notes.value = review.notes || "";
  notes.placeholder = "Optional notes";
  field("Notes", notes);

  const actions = document.createElement("div");
  actions.className = "review-actions";
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Save review";
  const message = document.createElement("span");
  message.className = "note";
  actions.append(save, message);
  form.append(actions);

  save.addEventListener("click", async () => {
    if (!fan.value || !compressor.value || !beep.value) {
      message.textContent = "Answer all three.";
      return;
    }
    save.disabled = true;
    message.textContent = "Saving…";
    const response = await fetch(`/api/events/${meta.id}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actualFan: fan.value === "yes",
        actualCompressor: compressor.value === "yes",
        actualBeep: beep.value === "yes",
        interference: [...tags.querySelectorAll("input:checked")]
          .map((input) => input.value),
        notes: notes.value,
      }),
    });
    save.disabled = false;
    message.textContent = response.ok
      ? "Review saved." : "Review could not be saved.";
    if (response.ok) {
      closeEventDetail();
      loadEvents();
    }
  });

  return form;
}

function buildReviewV2(meta) {
  if (meta.eventType === "manual") return buildReviewManual(meta);

  const form = document.createElement("div");
  form.className = "review-form";
  form.append(heading("Review"));

  const review = meta.review || {};
  let verdict = review.status === "reviewed" ? review.correct : null;
  const observation = meta.eventType === "fan" ||
    meta.eventType === "compressor";
  // Fan / compressor: what reality did around the reported X -> Y.
  let outcome = review.status === "reviewed"
    ? (review.outcome ?? (review.correct ? "correct" : null)) : null;

  const choices = document.createElement("div");
  choices.className = "review-actions";
  const correct = document.createElement("button");
  const wrong = document.createElement("button");
  const save = document.createElement("button");
  const message = document.createElement("span");
  correct.type = wrong.type = save.type = "button";
  correct.textContent = "Correct";
  wrong.textContent = "Wrong";
  save.textContent = "Save review";
  correct.className = wrong.className = "review-choice";
  message.className = "note";
  const stayedTrue = document.createElement("button");
  const stayedFalse = document.createElement("button");
  stayedTrue.type = stayedFalse.type = "button";
  stayedTrue.className = stayedFalse.className = "review-choice";
  stayedTrue.textContent = "No transition: true";
  stayedFalse.textContent = "No transition: false";
  choices.append(
    correct, ...(observation ? [stayedTrue, stayedFalse] : [wrong]),
    save, message,
  );
  form.append(choices);

  const field = (label, control) => {
    const wrapper = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    wrapper.append(name, control);
    form.append(wrapper);
  };

  const tags = document.createElement("div");
  tags.className = "review-tags";
  const selected = new Set(review.interference || []);
  for (const tag of ["talking", "printer", "tv", "other"]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = tag;
    input.checked = selected.has(tag);
    const text = document.createElement("span");
    text.textContent = tag;
    label.append(input, text);
    tags.append(label);
  }
  field("Interference", tags);

  const notes = document.createElement("textarea");
  notes.value = review.notes || "";
  notes.placeholder = "Optional notes";
  field("Notes", notes);

  const hint = document.createElement("p");
  hint.className = "note";
  hint.textContent = meta.eventType === "beep"
    ? "Correct: there really was a beep. Wrong: there was not."
    : observation
      ? `Correct: it really went ${meta.from} → ${meta.to}. ` +
        "No transition: true / false: the observation stayed that value " +
        "the whole time (so the detector was wrong to report a change). " +
        "Add detail in the notes if there is more to say."
      : "Correct: the capture is what you wanted.";
  form.append(hint);

  const showVerdict = () => {
    correct.classList.toggle(
      "active", observation ? outcome === "correct" : verdict === true,
    );
    wrong.classList.toggle("active", verdict === false);
    stayedTrue.classList.toggle("active", outcome === "stayed_true");
    stayedFalse.classList.toggle("active", outcome === "stayed_false");
  };
  correct.addEventListener("click", () => {
    verdict = true;
    outcome = "correct";
    showVerdict();
  });
  wrong.addEventListener("click", () => {
    verdict = false;
    showVerdict();
  });
  stayedTrue.addEventListener("click", () => {
    outcome = "stayed_true";
    showVerdict();
  });
  stayedFalse.addEventListener("click", () => {
    outcome = "stayed_false";
    showVerdict();
  });
  showVerdict();

  save.addEventListener("click", async () => {
    if (observation ? outcome === null : verdict === null) {
      message.textContent = observation
        ? "Choose Correct or a No transition."
        : "Choose Correct or Wrong.";
      return;
    }
    const body = {
      ...(observation ? { outcome } : { correct: verdict }),
      interference: [...tags.querySelectorAll("input:checked")]
        .map((input) => input.value),
      notes: notes.value,
    };

    save.disabled = true;
    message.textContent = "Saving…";
    const response = await fetch(`/api/events/${meta.id}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    save.disabled = false;
    message.textContent = response.ok
      ? "Review saved." : "Review could not be saved.";
    if (response.ok) {
      closeEventDetail();
      loadEvents();
    }
  });

  return form;
}

// The command the operator said they sent near an event, if any.
function buildCommandContext(meta) {
  const box = document.createElement("div");
  box.className = "command-context";
  box.append(heading("command context"));

  const reopen = () => openEvent(meta.id, document.querySelector("li.event.open"));

  if (meta.commandContext) {
    const c = meta.commandContext;
    box.append(table([
      ["command", c.command],
      ["expected a beep", c.expectedBeep ? "yes" : "no"],
      ["relative to this event",
        c.secondsFromEvent == null ? "set by hand"
          : `${c.secondsFromEvent > 0 ? "+" : ""}${c.secondsFromEvent} s`],
      ["note", c.note || "--"],
    ]));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      await fetch(`/api/events/${meta.id}/command-context`, { method: "DELETE" });
      reopen();
    });
    box.append(remove);
    return box;
  }

  const note = document.createElement("p");
  note.className = "note";
  note.textContent =
    "No command was marked within 10 s. If one was sent, attach it here.";
  const name = document.createElement("input");
  name.placeholder = "POWER";
  name.setAttribute("list", "commandNames");
  const expect = document.createElement("input");
  expect.type = "checkbox";
  expect.checked = true;
  const expectLabel = document.createElement("label");
  expectLabel.className = "inline";
  const expectText = document.createElement("span");
  expectText.textContent = "a beep is expected";
  expectLabel.append(expect, expectText);
  const attach = document.createElement("button");
  attach.type = "button";
  attach.textContent = "Attach";
  attach.addEventListener("click", async () => {
    if (!name.value.trim()) return;
    await fetch(`/api/events/${meta.id}/command-context`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        command: name.value.trim(),
        expectedBeep: expect.checked,
      }),
    });
    reopen();
  });
  box.append(note, name, expectLabel, attach);
  return box;
}

function buildReview(meta) {
  if (isV2(meta)) return buildReviewV2(meta);

  const form = document.createElement("div");
  form.className = "review-form";
  form.append(heading("Review"));

  let verdict = meta.review?.status === "reviewed"
    ? meta.review.classificationCorrect : null;

  const choices = document.createElement("div");
  choices.className = "review-actions";
  const correct = document.createElement("button");
  const wrong = document.createElement("button");
  const save = document.createElement("button");
  const message = document.createElement("span");
  correct.type = wrong.type = "button";
  save.type = "button";
  correct.textContent = "Correct";
  wrong.textContent = "Wrong";
  save.textContent = "Save review";
  correct.className = "review-choice";
  wrong.className = "review-choice";
  message.className = "note";
  choices.append(correct, wrong, save, message);
  form.append(choices);

  const stateSelect = (value) => {
    const select = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Select…";
    select.append(blank);
    for (const state of ["OFF", "FAN", "COMPRESSOR", "UNKNOWN"]) {
      const option = document.createElement("option");
      option.value = option.textContent = state;
      select.append(option);
    }
    select.value = value || "";
    return select;
  };

  const actualFrom = stateSelect(meta.review?.actualFrom);
  const actualTo = stateSelect(meta.review?.actualTo);
  const field = (label, control) => {
    const wrapper = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    wrapper.append(name, control);
    form.append(wrapper);
  };
  field("Actual from", actualFrom);
  field("Actual to", actualTo);

  const tags = document.createElement("div");
  tags.className = "review-tags";
  const selected = new Set(meta.review?.interference || []);
  for (const tag of ["talking", "printer", "tv", "other"]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = tag;
    input.checked = selected.has(tag);
    const text = document.createElement("span");
    text.textContent = tag;
    label.append(input, text);
    tags.append(label);
  }
  field("Interference", tags);

  const notes = document.createElement("textarea");
  notes.value = meta.review?.notes || "";
  notes.placeholder = "Optional notes";
  field("Notes", notes);

  const showVerdict = () => {
    correct.classList.toggle("active", verdict === true);
    wrong.classList.toggle("active", verdict === false);
  };
  correct.addEventListener("click", () => {
    verdict = true;
    actualFrom.value = meta.from || "UNKNOWN";
    actualTo.value = meta.to || "UNKNOWN";
    showVerdict();
  });
  wrong.addEventListener("click", () => {
    verdict = false;
    showVerdict();
  });
  showVerdict();

  save.addEventListener("click", async () => {
    if (verdict === null || !actualFrom.value || !actualTo.value) {
      message.textContent = "Choose Correct/Wrong and both actual states.";
      return;
    }
    save.disabled = true;
    message.textContent = "Saving…";
    const interference = [...tags.querySelectorAll("input:checked")]
      .map((input) => input.value);
    const response = await fetch(`/api/events/${meta.id}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classificationCorrect: verdict,
        actualFrom: actualFrom.value,
        actualTo: actualTo.value,
        interference,
        notes: notes.value,
      }),
    });
    save.disabled = false;
    message.textContent = response.ok ? "Review saved." : "Review could not be saved.";
    if (response.ok) {
      closeEventDetail();
      loadEvents();
    }
  });

  return form;
}

async function openEvent(id, item) {
  for (const other of document.querySelectorAll("li.event.open")) {
    other.classList.remove("open");
  }
  if (item) item.classList.add("open");

  const panel = $("detail");
  const body = $("detailBody");
  panel.hidden = false;
  releasePlaybackAudio(body);
  body.textContent = "loading…";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const meta = await (await fetch(`/api/events/${id}`)).json();

  body.textContent = "";

  const title = document.createElement("div");
  title.className = "state-name";
  title.style.fontSize = "24px";
  title.textContent = `Name: ${eventLabel(meta)}`;
  if (!isV2(meta)) title.classList.add("state-" + meta.to);
  body.append(title);

  body.append(heading("recording"));
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = `/api/events/${id}/audio`;
  audio.style.width = "100%";
  audio.style.marginTop = "14px";
  enablePlaybackGain(audio);
  body.append(audio);

  body.append(buildReview(meta));
  if (isV2(meta)) body.append(buildCommandContext(meta));

  body.append(heading("graph"));
  const chartLegend = document.createElement("div");
  chartLegend.className = "legend";
  const chart = document.createElement("div");
  chart.className = "chart";
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "computing…";
  body.append(chartLegend, chart, note);

  body.append(heading("full data"));
  const grid = document.createElement("div");
  grid.className = "grid2";
  const left = document.createElement("div");
  const observed = meta.observations || {};
  const rows = isV2(meta)
    ? [
      ["id", meta.id],
      ["event", eventLabel(meta)],
      ["source", meta.source || "transition"],
      ["time", meta.time.replace("T", " ")],
      ["fan at the time", yesNo(observed.fan)],
      ["compressor at the time", yesNo(observed.compressor)],
      ...(meta.eventType === "fan" || meta.eventType === "compressor"
        ? [["held before publishing", meta.candidateHeldSeconds + " s"]] : []),
      ...(meta.beep
        ? [
          ["beep duration", `${meta.beep.durationMs.toFixed(0)} ms`],
          ["beep pitch", `${meta.beep.peakHz.toFixed(1)} Hz`],
          ["beep contrast", `${meta.beep.contrastDb.toFixed(1)} dB`],
          ["beep level", `${meta.beep.levelDb.toFixed(1)} dB`],
        ] : []),
      ["classifier version", "V2"],
    ]
    : [
      ["id", meta.id],
      ["source", meta.source || "transition"],
      ["time", meta.time.replace("T", " ")],
      ["classifier", `${meta.from || "--"} → ${meta.to}`],
      // Events recorded before versions existed carry no version: v1.
      ["classifier version", (meta.classifierVersion || "v1").toUpperCase() +
        (meta.classifierVersion ? "" : " (recorded before versions)")],
      ["candidate held", meta.candidateHeldSeconds + " s"],
    ];
  rows.push(["audio", `${meta.audio.seconds} s ` +
    `(${meta.audio.preSeconds} before / ${meta.audio.postSeconds} after)`]);
  left.append(heading(isV2(meta) ? "event" : "transition"), table(rows));
  const recorded = meta.featuresAtTransition || {};
  left.append(heading("stationarity at transition"), table(
    ["1k-2k_std", "500-1k_std", "2k-4k_std"].map((name) => [
      name,
      recorded[name] == null
        ? "not recorded"
        : formatFeatureValue(name, recorded[name]),
    ]),
  ));
  left.append(heading("features at transition"), table(
    Object.entries(meta.featuresAtTransition).map(
      ([k, v]) => [k, formatFeatureValue(k, v)]),
  ));

  const right = document.createElement("div");
  right.append(heading("classifier config at the time"), table(
    Object.entries(meta.classifierConfig).map(([k, v]) => [k, v]),
  ));
  const stats = document.createElement("table");
  stats.className = "kv";
  const head = stats.createTHead().insertRow();
  for (const label of ["feature", "median", "min", "max", "std"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  for (const [feature, summary] of Object.entries(meta.decisionWindow || {})) {
    const row = stats.insertRow();
    row.insertCell().textContent = feature;
    for (const key of ["median", "min", "max", "std"]) {
      row.insertCell().textContent = summary[key];
    }
  }
  right.append(heading("decision window"), stats);
  grid.append(left, right);
  body.append(grid);

  const detailActions = document.createElement("div");
  detailActions.className = "detail-actions";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "danger";
  remove.textContent = "Delete event";
  remove.addEventListener("click", async () => {
    if (!confirm("Delete this event's JSON and WAV permanently?")) return;
    remove.disabled = true;
    const response = await fetch(`/api/events/${meta.id}`, { method: "DELETE" });
    if (!response.ok) {
      remove.disabled = false;
      remove.textContent = "Delete failed";
      return;
    }
    closeEventDetail();
    selectedEventIds.delete(meta.id);
    if (eventPage > 1 && $("events").children.length === 1) eventPage -= 1;
    loadEvents();
  });
  detailActions.append(remove);
  body.append(detailActions);

  const timeline = await (
    await fetch(`/api/events/${id}/timeline`)
  ).json();

  const c = timeline.columns;
  legend(chartLegend, ["rms", ...BANDS]);

  const v2Timeline = timeline.classifierVersion === "v2";
  const beeps = timeline.beeps || [];
  const commandMarks = (timeline.commands || []).map((cmd) => ({
    t: cmd.t, colour: "#d6a8f0", label: cmd.command,
  }));

  lineChart(chart, c.t, [
    { name: "rms", values: c.rms },
    ...BANDS.map((b) => ({ name: b, values: c[b] })),
  ], v2Timeline ? {
    height: 240,
    strips: [
      { name: "fan", values: c.fan, colour: "#1f9d55" },
      { name: "comp", values: c.compressor, colour: "#d2691e" },
    ],
    marks: [...beeps.map((b) => ({ t: b.startSeconds })), ...commandMarks],
    zeroLine: true,
    xFormat: (v) => v.toFixed(0) + "s",
  } : {
    height: 220,
    states: c.state,
    marks: commandMarks,
    zeroLine: true,
    xFormat: (v) => v.toFixed(0) + "s",
  });

  if (v2Timeline && meta.eventType === "beep") {
    body.insertBefore(heading("beep contrast"), body.querySelector(".grid2")
      .previousElementSibling);
    const contrastChart = document.createElement("div");
    contrastChart.className = "chart";
    body.insertBefore(contrastChart, body.querySelector(".grid2")
      .previousElementSibling);
    lineChart(contrastChart, c.t, [
      { name: "contrast", values: c.beepContrast, colour: "#e0c341" },
    ], {
      height: 120,
      marks: [...beeps.map((b) => ({ t: b.startSeconds })), ...commandMarks],
      zeroLine: true,
      xFormat: (v) => v.toFixed(0) + "s",
    });
  }

  note.textContent = v2Timeline
    ? `${timeline.count} windows, recomputed from the WAV with the ` +
      `settings recorded in the event. The two strips are the published ` +
      `fan and compressor observations, independently; yellow ticks are ` +
      `beeps found in the audio, purple ticks are commands marked during ` +
      `the clip. Near the left edge the strips are still ` +
      `warming up, because this replay starts cold.`
    : 
    `${timeline.count} windows, recomputed from the WAV with the ` +
    `settings recorded in the event. The strip is the published ` +
    `state; near the left edge it is still warming up, because this ` +
    `replay starts cold while the live classifier had history from ` +
    `before the pre-roll.`;
}

$("closeDetail").addEventListener("click", closeEventDetail);

loadHistory();
setInterval(loadHistory, 15000);
