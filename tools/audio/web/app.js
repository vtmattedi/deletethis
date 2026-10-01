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

const $ = (id) => document.getElementById(id);
const DEFAULT_PLAYBACK_GAIN_DB = 6;

let lastEventCount = -1;
let settingsDirty = false;
let eventPage = 1;
let eventPages = 0;
const EVENT_PAGE_SIZE = 10;
const selectedEventIds = new Set();
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
  for (const [key, , kind] of SETTINGS) {
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
  for (const [name] of SETTINGS) {
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
    for (const [name] of SETTINGS) {
      const field = $("cfg-" + name);
      if (field) field.value = config[name];
    }
  }
}

$("saveSettings").addEventListener("click", saveSettings);

function render(data) {
  const state = data.state || "--";
  const element = $("state");
  element.textContent = state;
  element.className = "state-name state-" + state;

  $("candidate").textContent = data.candidate || "--";
  $("stable").textContent =
    data.stableSeconds == null ? "--" : data.stableSeconds.toFixed(1) + " s";

  renderFeatures(data.features || {});

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

    const what = document.createElement("strong");
    what.textContent = `${event.from || "--"} → ${event.to}`;

    const review = document.createElement("span");
    let correctness = event.review?.status === "reviewed"
      ? (event.review.classificationCorrect ? "correct" : "incorrect")
      : "unreviewed";
    review.className = `review-status review-${correctness}`;
    review.textContent = correctness;

    const quickCorrect = document.createElement("button");
    quickCorrect.type = "button";
    quickCorrect.className = "quick-correct";
    quickCorrect.textContent = correctness === "correct" ? "Correct ✓" : "Correct";
    quickCorrect.title = "Mark correct with no interference or notes";
    quickCorrect.disabled = correctness === "correct";
    quickCorrect.addEventListener("click", async () => {
      quickCorrect.disabled = true;
      quickCorrect.textContent = "Saving…";
      try {
        const response = await fetch(`/api/events/${event.id}/review`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            classificationCorrect: true,
            actualFrom: event.from || "UNKNOWN",
            actualTo: event.to || "UNKNOWN",
            interference: [],
            notes: "",
          }),
        });
        if (!response.ok) throw new Error("review request failed");

        const updated = await response.json();
        event.review = updated.review;
        correctness = "correct";
        review.className = "review-status review-correct";
        review.textContent = "correct";
        quickCorrect.textContent = "Correct ✓";
        if (item.classList.contains("open")) openEvent(event.id, item);
      } catch (_) {
        quickCorrect.disabled = false;
        quickCorrect.textContent = "Retry correct";
      }
    });

    const held = document.createElement("span");
    held.className = "note";
    held.textContent =
      `${event.source === "manual" ? "manual · " : ""}` +
      `held ${event.candidateHeldSeconds}s · ` +
      `${event.audio.seconds}s audio`;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/events/${event.id}/audio`;
    enablePlaybackGain(audio);

    item.append(select, when, what, review, quickCorrect, held, audio);
    list.append(item);
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

  const panel = $("bulkReview");
  const body = $("bulkReviewBody");
  $("bulkActionNote").textContent = "";
  $("bulkReviewSummary").textContent =
    `This review will be applied to ${identifiers.length} selected event` +
    `${identifiers.length === 1 ? "" : "s"}. Correct uses each event's ` +
    "own classifier transition.";
  body.textContent = "";

  const form = document.createElement("div");
  form.className = "review-form";
  let verdict = null;

  const choices = document.createElement("div");
  choices.className = "review-actions";
  const correct = document.createElement("button");
  const wrong = document.createElement("button");
  const save = document.createElement("button");
  const message = document.createElement("span");
  correct.type = wrong.type = save.type = "button";
  correct.textContent = "Correct";
  wrong.textContent = "Wrong";
  save.textContent = "Save reviews";
  correct.className = wrong.className = "review-choice";
  message.className = "note";
  choices.append(correct, wrong, save, message);
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

  const actualFrom = stateSelect();
  const actualTo = stateSelect();
  field("Actual from", actualFrom);
  field("Actual to", actualTo);

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
    actualFrom.disabled = verdict === true;
    actualTo.disabled = verdict === true;
    if (verdict === true) {
      actualFrom.value = "";
      actualTo.value = "";
    }
  };
  correct.addEventListener("click", () => {
    verdict = true;
    showVerdict();
    message.textContent = "Each event will use its own classifier labels.";
  });
  wrong.addEventListener("click", () => {
    verdict = false;
    showVerdict();
    message.textContent = "";
  });
  showVerdict();

  save.addEventListener("click", async () => {
    if (verdict === null) {
      message.textContent = "Choose Correct or Wrong.";
      return;
    }
    if (verdict === false && (!actualFrom.value || !actualTo.value)) {
      message.textContent = "Choose both actual states for an incorrect review.";
      return;
    }

    const payload = {
      ids: identifiers,
      classificationCorrect: verdict,
      interference: [...tags.querySelectorAll("input:checked")]
        .map((input) => input.value),
      notes: notes.value,
    };
    if (verdict === false) {
      payload.actualFrom = actualFrom.value;
      payload.actualTo = actualTo.value;
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
      message.textContent = "Reviews could not be saved. Refresh and retry.";
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

function buildReview(meta) {
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
  title.textContent = `Name: ${meta.from || "--"} → ${meta.to}`;
  title.classList.add("state-" + meta.to);
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
  left.append(heading("transition"), table([
    ["id", meta.id],
    ["source", meta.source || "transition"],
    ["time", meta.time.replace("T", " ")],
    ["classifier", `${meta.from || "--"} → ${meta.to}`],
    ["candidate held", meta.candidateHeldSeconds + " s"],
    ["audio", `${meta.audio.seconds} s ` +
      `(${meta.audio.preSeconds} before / ${meta.audio.postSeconds} after)`],
  ]));
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

$("closeDetail").addEventListener("click", closeEventDetail);

loadHistory();
setInterval(loadHistory, 15000);
