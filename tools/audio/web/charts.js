// Small SVG line/strip charts.
//
// Hand-rolled rather than pulled from a CDN: this is a local debug
// tool that should work with no network, and the whole requirement is
// "a few dB traces and a state strip".

const NS = "http://www.w3.org/2000/svg";

const STATE_COLOUR = {
  OFF: "#3b7dd8",
  FAN: "#1f9d55",
  COMPRESSOR: "#d2691e",
};

const SERIES_COLOUR = {
  "rms": "#e7e9ee",
  "30-80": "#d2691e",
  "500-1k": "#1f9d55",
  "1k-2k": "#3b7dd8",
  "200-1200": "#9b7ede",
};

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function niceBounds(values) {
  const clean = values.filter((v) => v != null && isFinite(v));
  if (!clean.length) return [-120, 0];

  let lo = Math.min(...clean);
  let hi = Math.max(...clean);

  if (hi - lo < 6) {
    const mid = (hi + lo) / 2;
    lo = mid - 3;
    hi = mid + 3;
  }

  const pad = (hi - lo) * 0.08;
  return [Math.floor(lo - pad), Math.ceil(hi + pad)];
}

/**
 * @param host      container element, emptied first
 * @param t         x values (seconds, any origin)
 * @param series    [{name, values}]
 * @param options   {height, states, strips, marks, zeroLine, outages}
 *                  strips: [{name, values (true/false/null), colour}] one
 *                  thin on/off strip per independent observation;
 *                  marks: [{t, colour}] vertical ticks, for beeps.
 */
export function lineChart(host, t, series, options = {}) {
  host.textContent = "";

  const width = Math.max(host.clientWidth || 600, 320);
  const height = options.height || 170;
  const left = 46;
  const right = 10;
  const top = 8;
  const STRIP = 9;
  const strips = options.strips || [];
  const stripHeight = options.states
    ? 14
    : strips.length * (STRIP + 3);
  const bottom = 20 + stripHeight;

  const plotW = width - left - right;
  const plotH = height - top - bottom;

  const svg = el("svg", {
    width: "100%",
    height,
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none",
  });

  if (!t.length) {
    const empty = el("text", {
      x: width / 2, y: height / 2, "text-anchor": "middle",
      fill: "#949bab", "font-size": "12",
    });
    empty.textContent = options.emptyText || "no data yet";
    svg.append(empty);
    host.append(svg);
    return;
  }

  const t0 = t[0];
  const t1 = t[t.length - 1];
  const span = t1 - t0 || 1;

  const all = series.flatMap((s) => s.values);
  const [lo, hi] = options.bounds || niceBounds(all);

  const x = (v) => left + ((v - t0) / span) * plotW;
  const y = (v) => top + plotH - ((v - lo) / (hi - lo)) * plotH;

  // grid + y labels
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4;
    const yy = y(value);

    svg.append(el("line", {
      x1: left, x2: left + plotW, y1: yy, y2: yy,
      stroke: "#2c313d", "stroke-width": 1,
    }));

    const label = el("text", {
      x: left - 6, y: yy + 3, "text-anchor": "end",
      fill: "#949bab", "font-size": "10",
    });
    label.textContent = value.toFixed(0);
    svg.append(label);
  }

  // Outages: the stream was down, so there is no data here rather
  // than data saying zero. Shade them so a flat gap is not read as a
  // quiet room.
  for (const [from, to] of options.outages || []) {
    const x0 = Math.max(x(from), left);
    const x1 = Math.min(to == null ? left + plotW : x(to), left + plotW);

    if (x1 <= x0) continue;

    svg.append(el("rect", {
      x: x0, y: top, width: x1 - x0, height: plotH,
      fill: "#d64545", opacity: 0.16,
    }));
  }

  // x = 0 marker (used by the event view for the transition)
  if (options.zeroLine && t0 <= 0 && t1 >= 0) {
    svg.append(el("line", {
      x1: x(0), x2: x(0), y1: top, y2: top + plotH,
      stroke: "#e7e9ee", "stroke-width": 1.5,
      "stroke-dasharray": "4 3", opacity: 0.8,
    }));

    const mark = el("text", {
      x: x(0) + 4, y: top + 10, fill: "#e7e9ee", "font-size": "10",
    });
    mark.textContent = "transition";
    svg.append(mark);
  }

  for (const s of series) {
    let path = "";
    let pen = false;

    for (let i = 0; i < t.length; i++) {
      const v = s.values[i];

      if (v == null || !isFinite(v)) { pen = false; continue; }

      path += `${pen ? "L" : "M"}${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    }

    svg.append(el("path", {
      d: path, fill: "none",
      stroke: s.colour || SERIES_COLOUR[s.name] || "#888",
      "stroke-width": s.width || 1.4,
      "stroke-linejoin": "round",
    }));
  }

  // state strip along the bottom
  if (options.states) {
    const stripY = top + plotH + 6;
    let runStart = 0;

    const flush = (endIndex) => {
      const state = options.states[runStart];
      if (state) {
        const x0 = x(t[runStart]);
        const x1 = x(t[Math.min(endIndex, t.length - 1)]);
        svg.append(el("rect", {
          x: x0, y: stripY, width: Math.max(1, x1 - x0),
          height: stripHeight, fill: STATE_COLOUR[state] || "#4a5162",
        }));
      }
      runStart = endIndex;
    };

    for (let i = 1; i < t.length; i++) {
      if (options.states[i] !== options.states[runStart]) flush(i);
    }
    flush(t.length - 1);
  }

  // One on/off strip per observation: nothing here says what the
  // observations add up to.
  strips.forEach((strip, row) => {
    const stripY = top + plotH + 6 + row * (STRIP + 3);
    let runStart = 0;

    const flush = (endIndex) => {
      if (strip.values[runStart]) {
        const x0 = x(t[runStart]);
        const x1 = x(t[Math.min(endIndex, t.length - 1)]);
        svg.append(el("rect", {
          x: x0, y: stripY, width: Math.max(1, x1 - x0),
          height: STRIP, fill: strip.colour || "#4a5162",
        }));
      }
      runStart = endIndex;
    };

    for (let i = 1; i < t.length; i++) {
      if (strip.values[i] !== strip.values[runStart]) flush(i);
    }
    flush(t.length - 1);

    const name = el("text", {
      x: left - 6, y: stripY + STRIP - 1, "text-anchor": "end",
      fill: "#949bab", "font-size": "8",
    });
    name.textContent = strip.name;
    svg.append(name);
  });

  for (const mark of options.marks || []) {
    if (mark.t < t0 || mark.t > t1) continue;

    svg.append(el("line", {
      x1: x(mark.t), x2: x(mark.t), y1: top, y2: top + plotH,
      stroke: mark.colour || "#e0c341", "stroke-width": 1.5,
      opacity: 0.9,
    }));
  }

  // x labels
  const xLabel = (value, anchor, px) => {
    const label = el("text", {
      x: px, y: height - 6, "text-anchor": anchor,
      fill: "#949bab", "font-size": "10",
    });
    label.textContent = value;
    svg.append(label);
  };

  const fmt = options.xFormat || ((v) => v.toFixed(0) + "s");
  xLabel(fmt(t0), "start", left);
  xLabel(fmt(t1), "end", left + plotW);

  host.append(svg);
}

export function legend(host, names) {
  host.textContent = "";

  for (const name of names) {
    const item = document.createElement("span");
    item.className = "key";

    const swatch = document.createElement("i");
    swatch.style.background = SERIES_COLOUR[name] || "#888";

    item.append(swatch, document.createTextNode(name));
    host.append(item);
  }
}

export { STATE_COLOUR };
