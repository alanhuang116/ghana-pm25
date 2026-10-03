/* Colour scales, thresholds and grid decoding for GH-PM25. */
(function (global) {
  "use strict";

  // Semantic-heat sequential ramp with monotonically decreasing lightness.
  // Stops are placed at policy-relevant concentrations (ug/m3).
  const PM_STOPS = [
    [0, "#fff7d6"], [5, "#fde9a6"], [15, "#fcc868"], [25, "#f8a24a"], [35, "#ef7a3a"],
    [55, "#d9502f"], [75, "#b32e32"], [100, "#861e3b"], [150, "#58163f"], [250, "#2c0c2e"],
  ];
  const UNC_STOPS = [[0, "#e7f0fb"], [0.5, "#9ec5f4"], [1.0, "#5598e7"], [2.0, "#1c5cab"], [4.0, "#0d366b"]];
  // Diverging blue <-> red with neutral grey midpoint (difference to CAMS)
  const DIFF_STOPS = [[-60, "#104281"], [-25, "#3987e5"], [-8, "#b7d3f6"], [0, "#f0efec"], [8, "#f6c1b8"], [25, "#e34948"], [60, "#8a1c1c"]];

  // US EPA AQI (PM2.5, 24-h, 2024 revision) - official category colours
  const AQI = [
    { max: 9.0, label: "Good", color: "#00e400", aqi: [0, 50] },
    { max: 35.4, label: "Moderate", color: "#ffff00", aqi: [51, 100] },
    { max: 55.4, label: "Unhealthy for sensitive groups", color: "#ff7e00", aqi: [101, 150] },
    { max: 125.4, label: "Unhealthy", color: "#ff0000", aqi: [151, 200] },
    { max: 225.4, label: "Very unhealthy", color: "#8f3f97", aqi: [201, 300] },
    { max: Infinity, label: "Hazardous", color: "#7e0023", aqi: [301, 500] },
  ];
  const AQI_BP = [[0, 9.0, 0, 50], [9.1, 35.4, 51, 100], [35.5, 55.4, 101, 150], [55.5, 125.4, 151, 200], [125.5, 225.4, 201, 300], [225.5, 325.4, 301, 500]];

  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

  function buildLUT(stops, vmin, vmax, n = 1024) {
    const s = stops.map(([v, c]) => [v, hexToRgb(c)]);
    const lut = new Uint8ClampedArray(n * 3);
    for (let i = 0; i < n; i++) {
      const v = vmin + (vmax - vmin) * i / (n - 1);
      let k = 0;
      while (k < s.length - 2 && v > s[k + 1][0]) k++;
      const [v0, c0] = s[k], [v1, c1] = s[k + 1];
      const t = Math.max(0, Math.min(1, (v - v0) / (v1 - v0)));
      for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * t;
    }
    return { lut, vmin, vmax, n };
  }

  // piecewise scale position (so that legend ticks are evenly spaced at stops)
  function makePiecewise(stops) {
    const vals = stops.map(s => s[0]);
    return {
      pos(v) { // 0..1
        if (v <= vals[0]) return 0;
        for (let i = 0; i < vals.length - 1; i++) if (v <= vals[i + 1]) return (i + (v - vals[i]) / (vals[i + 1] - vals[i])) / (vals.length - 1);
        return 1;
      },
      stops, vals,
    };
  }

  function colorAt(stops, v) {
    const pw = makePiecewise(stops);
    const p = pw.pos(v) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(p)), t = p - i;
    const a = hexToRgb(stops[i][1]), b = hexToRgb(stops[i + 1][1]);
    return `rgb(${a.map((x, j) => Math.round(x + (b[j] - x) * t)).join(",")})`;
  }

  function aqiCategory(pm) {
    for (const c of AQI) if (pm <= c.max) return c;
    return AQI[AQI.length - 1];
  }
  function aqiValue(pm) {
    const c = Math.floor(pm * 10) / 10;
    for (const [cl, ch, il, ih] of AQI_BP) if (c <= ch) return Math.round((ih - il) / (ch - cl) * (c - cl) + il);
    return 500;
  }

  // Decode PNG grids (see ghana_pm25/export.py)
  async function loadPng(url) {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, c.width, c.height);
  }

  async function loadDay(base, dateStr) {
    const y = dateStr.slice(0, 4), d = dateStr.replaceAll("-", "");
    const [main, aux] = await Promise.all([loadPng(`${base}/grids/${y}/${d}_pm25.png`), loadPng(`${base}/grids/${y}/${d}_aux.png`)]);
    const w = main.width, h = main.height, n = w * h;
    const pm = new Float32Array(n), lo = new Float32Array(n), hi = new Float32Array(n), cams = new Float32Array(n), code = new Uint8Array(n);
    const M = main.data, A = aux.data;
    for (let i = 0; i < n; i++) {
      const k = i * 4;
      code[i] = M[k + 2];
      if (code[i] === 0) { pm[i] = NaN; lo[i] = NaN; hi[i] = NaN; cams[i] = NaN; continue; }
      const v = (M[k] * 256 + M[k + 1]) / 10;
      pm[i] = v; lo[i] = v * A[k] / 250; hi[i] = v * A[k + 1] / 40; cams[i] = A[k + 2];
    }
    return { w, h, pm, lo, hi, cams, code, date: dateStr };
  }

  global.GHScales = { PM_STOPS, UNC_STOPS, DIFF_STOPS, AQI, buildLUT, makePiecewise, colorAt, aqiCategory, aqiValue, loadDay, hexToRgb };
})(window);
