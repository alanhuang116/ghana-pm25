/* GH-PM25 Observatory - main application */
(function () {
  "use strict";
  const S = window.GHScales, C = window.GHCharts;
  const DATA = "data";
  const NORTH = ["Northern", "Savannah", "North East", "Upper East", "Upper West"].flatMap(n => [n, `${n} Region`]);
  const state = {
    manifest: null, date: null, day: null, dayCache: new Map(), layer: "pm25", stations: [], series: {}, regions: null,
    cubes: new Map(), map: null, playing: null, selPoint: null, validation: null, obsByDate: new Map(), themeDark: false,
  };

  const $ = (id) => document.getElementById(id);
  const fetchJSON = (p) => fetch(`${DATA}/${p}`, { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error(p); return r.json(); });
  const isDark = () => document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);

  /* ============================== bootstrap ============================== */
  async function init() {
    setupTabs();
    setupTheme();
    try {
      state.manifest = await fetchJSON("manifest.json");
    } catch (e) {
      $("map-note").textContent = "No products found in portal/data — run the pipeline (scripts/run_daily.py).";
      return;
    }
    const m = state.manifest;
    $("foot-version").textContent = `${m.product} v${m.version} · generated ${m.generated_at.replace("T", " ").slice(0, 16)} UTC`;
    renderStatus();
    const [stations, series, regions] = await Promise.all([
      fetchJSON("stations/stations.json"), fetchJSON("stations/series.json").catch(() => ({})), fetchJSON("series/regions.json").catch(() => null)]);
    state.stations = stations; state.series = series; state.regions = regions;
    indexObservations();
    initMap();
    initControls();
    initStationsTab();
    initExposureTab();
    initValidationTab();
    renderMethods();
    await setDate(m.latest);
    setInterval(checkForUpdates, 10 * 60 * 1000);
  }

  function setupTabs() {
    document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-selected", x === b ? "true" : "false"));
      document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
      if (b.dataset.tab === "monitor" && state.map) setTimeout(() => state.map.resize(), 50);
      rerenderTab(b.dataset.tab);
    }));
  }
  function rerenderTab(tab) {
    if (tab === "exposure") renderExposure();
    if (tab === "validation") renderValidation();
    if (tab === "stations" && state.selStation) renderStation(state.selStation);
  }

  function setupTheme() {
    try { const t = localStorage.getItem("ghpm25-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    $("theme-toggle").onclick = () => {
      const next = isDark() ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("ghpm25-theme", next); } catch (e) { /* ignore */ }
      if (state.map) setBasemap();
      rerenderAll();
    };
  }
  function rerenderAll() {
    const active = document.querySelector(".tabs button[aria-selected=true]").dataset.tab;
    if (active === "monitor") { renderSeriesPanel(); }
    rerenderTab(active);
  }

  function renderStatus() {
    const m = state.manifest;
    const ageH = (Date.now() - Date.parse(m.generated_at + "Z")) / 36e5;
    const color = ageH < 30 ? "var(--good)" : ageH < 72 ? "var(--warning)" : "var(--critical)";
    const label = ageH < 30 ? "Operational" : ageH < 72 ? "Delayed" : "Stale";
    $("run-status").innerHTML = `<span class="dot" style="background:${color}"></span>${label} · last run ${Math.round(ageH)} h ago`;
    $("run-status").title = Object.entries(m.freshness || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
  }

  async function checkForUpdates() {
    try {
      const m = await fetchJSON("manifest.json");
      if (m.generated_at !== state.manifest.generated_at) {
        const wasLatest = state.date === state.manifest.latest;
        state.manifest = m; state.dayCache.clear();
        renderStatus();
        if (wasLatest) await setDate(m.latest);
      }
    } catch (e) { /* offline */ }
  }

  function indexObservations() {
    for (const [sid, s] of Object.entries(state.series)) {
      (s.date || []).forEach((d, i) => {
        if (!state.obsByDate.has(d)) state.obsByDate.set(d, []);
        state.obsByDate.get(d).push({ site_id: sid, pm25: s.pm25[i], raw: s.pm25_raw ? s.pm25_raw[i] : null });
      });
    }
  }

  /* ============================== map ============================== */
  function basemapTiles(kind) {
    // Esri World Light/Dark Gray Canvas (keyless, attribution required)
    const tone = isDark() ? "Dark" : "Light";
    const svc = kind === "nolabels" ? `World_${tone}_Gray_Base` : `World_${tone}_Gray_Reference`;
    return [`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${svc}/MapServer/tile/{z}/{y}/{x}`];
  }
  function initMap() {
    const g = state.manifest.grid;
    const map = new maplibregl.Map({
      container: "map", attributionControl: { compact: true },
      style: { version: 8, sources: {}, layers: [] },
      bounds: [[g.lon0 - 0.3, g.lat1 - 6.8], [g.lon0 + 4.85, g.lat1 + 0.3]], fitBoundsOptions: { padding: 10 }, maxZoom: 13, minZoom: 4,
    });
    state.map = map;
    window.__ghmap = map;  // exposed for automated QA
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("load", async () => {
      map.addSource("base", { type: "raster", tiles: basemapTiles("nolabels"), tileSize: 256, maxzoom: 16, attribution: "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors" });
      map.addLayer({ id: "base", type: "raster", source: "base" });
      const canvas = document.createElement("canvas"); canvas.width = g.w; canvas.height = g.h;
      state.canvas = canvas;
      map.addSource("pm", { type: "image", url: canvas.toDataURL(), coordinates: [[g.lon0, g.lat1], [g.lon1, g.lat1], [g.lon1, g.lat0], [g.lon0, g.lat0]] });
      map.addLayer({ id: "pm", type: "raster", source: "pm", paint: { "raster-opacity": +$("opacity").value, "raster-resampling": "nearest", "raster-fade-duration": 0 } });
      map.addSource("labels", { type: "raster", tiles: basemapTiles("only_labels"), tileSize: 256 });
      map.addLayer({ id: "labels", type: "raster", source: "labels" });
      for (const lvl of [0, 1, 2]) {
        const gj = await fetchJSON(`boundaries/adm${lvl}.geojson`).catch(() => null);
        if (!gj) continue;
        map.addSource(`adm${lvl}`, { type: "geojson", data: gj });
        map.addLayer({ id: `adm${lvl}-fill`, type: "fill", source: `adm${lvl}`, paint: { "fill-opacity": 0 } });
        map.addLayer({ id: `adm${lvl}-line`, type: "line", source: `adm${lvl}`, paint: { "line-color": isDark() ? "#d9d8d2" : "#3d3c39", "line-width": lvl === 0 ? 1.4 : lvl === 1 ? 0.9 : 0.4, "line-opacity": lvl === 2 ? 0.55 : 0.8 },
          layout: { visibility: lvl === 0 || String(lvl) === $("adm-select").value ? "visible" : "none" } });
      }
      map.addSource("stations", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: "stations", type: "circle", source: "stations", paint: {
        "circle-radius": ["case", ["get", "ref"], 7, 5], "circle-color": ["get", "color"], "circle-stroke-color": isDark() ? "#1a1a19" : "#ffffff",
        "circle-stroke-width": 2, "circle-opacity": ["case", ["get", "has"], 1, 0.35] } });
      map.on("mousemove", onMapMove);
      map.on("mouseout", () => { $("readout").hidden = true; });
      map.on("click", onMapClick);
      map.on("mouseenter", "stations", () => map.getCanvas().style.cursor = "pointer");
      map.on("mouseleave", "stations", () => map.getCanvas().style.cursor = "");
      state.mapReady = true;
      drawLayer();
      map.once("idle", () => console.log("[ghpm25] idle; rendered stations:", map.queryRenderedFeatures({ layers: ["stations"] }).length,
        "adm1 lines:", map.queryRenderedFeatures({ layers: ["adm1-line"] }).length));
    });
  }
  function setBasemap() {
    const map = state.map; if (!state.mapReady) return;
    map.getSource("base").setTiles(basemapTiles("nolabels"));
    map.getSource("labels").setTiles(basemapTiles("only_labels"));
    for (const lvl of [0, 1, 2]) if (map.getLayer(`adm${lvl}-line`)) map.setPaintProperty(`adm${lvl}-line`, "line-color", isDark() ? "#d9d8d2" : "#3d3c39");
    map.setPaintProperty("stations", "circle-stroke-color", isDark() ? "#1a1a19" : "#ffffff");
  }

  function layerValue(i) {
    const d = state.day; if (!d) return NaN;
    switch (state.layer) {
      case "pm25": case "aqi": case "who": case "gepa": return d.pm[i];
      case "lower": return d.lo[i];
      case "upper": return d.hi[i];
      case "relunc": return (d.hi[i] - d.lo[i]) / d.pm[i];
      case "cams": return d.cams[i];
      case "diff": return d.pm[i] - d.cams[i];
    }
    return NaN;
  }

  function colorFor(v) {
    const L = state.layer;
    if (!Number.isFinite(v)) return null;
    if (L === "aqi") return S.hexToRgb(S.aqiCategory(v).color);
    if (L === "who") return v > 15 ? [217, 80, 47] : [205, 226, 251];
    if (L === "gepa") return v > 35 ? [134, 30, 59] : [205, 226, 251];
    const stops = L === "relunc" ? S.UNC_STOPS : L === "diff" ? S.DIFF_STOPS : S.PM_STOPS;
    const pw = state.pw && state.pwStops === stops ? state.pw : (state.pw = S.makePiecewise(stops), state.pwStops = stops, state.pw);
    const lut = state.lut && state.lutStops === stops ? state.lut : (state.lut = buildPosLUT(stops), state.lutStops = stops, state.lut);
    const k = Math.round(pw.pos(v) * 1023) * 3;
    return [lut[k], lut[k + 1], lut[k + 2]];
  }
  function buildPosLUT(stops) {
    const n = 1024, lut = new Uint8ClampedArray(n * 3), m = stops.length - 1;
    for (let i = 0; i < n; i++) {
      const p = i / (n - 1) * m, j = Math.min(m - 1, Math.floor(p)), t = p - j;
      const a = S.hexToRgb(stops[j][1]), b = S.hexToRgb(stops[j + 1][1]);
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[c] + (b[c] - a[c]) * t;
    }
    return lut;
  }

  function drawLayer() {
    if (!state.mapReady || !state.day) return;
    const g = state.manifest.grid, d = state.day;
    const ctx = state.canvas.getContext("2d");
    const img = ctx.createImageData(d.w, d.h);
    // row-wise reprojection from the regular lat grid to Web Mercator rows
    const merc = (lat) => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
    const y0 = merc(g.lat1), y1 = merc(g.lat0);
    const maskAoa = $("chk-aoa").checked;
    for (let r = 0; r < d.h; r++) {
      const ym = y0 + (y1 - y0) * (r + 0.5) / d.h;
      const lat = (2 * Math.atan(Math.exp(ym)) - Math.PI / 2) * 180 / Math.PI;
      const sr = Math.min(d.h - 1, Math.max(0, Math.floor((g.lat1 - lat) / g.res)));
      for (let c = 0; c < d.w; c++) {
        const i = sr * d.w + c, o = (r * d.w + c) * 4;
        if (d.code[i] === 0 || (maskAoa && d.code[i] === 1)) { img.data[o + 3] = 0; continue; }
        const col = colorFor(layerValue(i));
        if (!col) { img.data[o + 3] = 0; continue; }
        img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    state.map.getSource("pm").updateImage({ url: state.canvas.toDataURL(), coordinates: [[g.lon0, g.lat1], [g.lon1, g.lat1], [g.lon1, g.lat0], [g.lon0, g.lat0]] });
    renderLegend();
    updateStationsLayer();
  }

  function renderLegend() {
    const L = state.layer, el = $("legend");
    const titles = { pm25: "PM2.5 daily mean (µg/m³)", lower: "90% lower bound (µg/m³)", upper: "90% upper bound (µg/m³)", cams: "CAMS PM2.5 (µg/m³)", relunc: "Relative 90% interval width", diff: "GH-PM25 − CAMS (µg/m³)" };
    if (L === "aqi") {
      el.innerHTML = `<div class="title">US EPA AQI category (PM2.5, 24-h)</div><div class="cats">` +
        S.AQI.map((c, i) => `<i class="sw" style="background:${c.color}"></i><span>${c.label} (${i === 0 ? "0" : S.AQI[i - 1].max + 0.1}–${Number.isFinite(c.max) ? c.max : "+"})</span>`).join("") + "</div>";
      return;
    }
    if (L === "who" || L === "gepa") {
      const t = L === "who" ? 15 : 35;
      el.innerHTML = `<div class="title">${L === "who" ? "WHO 2021 24-h guideline" : "Ghana EPA 24-h standard"} (${t} µg/m³)</div><div class="cats"><i class="sw" style="background:rgb(205,226,251)"></i><span>≤ ${t}</span><i class="sw" style="background:${L === "who" ? "rgb(217,80,47)" : "rgb(134,30,59)"}"></i><span>&gt; ${t} — exceedance</span></div>`;
      return;
    }
    const stops = L === "relunc" ? S.UNC_STOPS : L === "diff" ? S.DIFF_STOPS : S.PM_STOPS;
    const grad = stops.map((s, i) => `${s[1]} ${(i / (stops.length - 1) * 100).toFixed(1)}%`).join(",");
    const ticks = stops.map((s, i) => `<span style="left:${(i / (stops.length - 1) * 100).toFixed(1)}%">${s[0]}</span>`).join("");
    el.innerHTML = `<div class="title">${titles[L]}</div><div class="ramp" style="background:linear-gradient(90deg,${grad})"></div><div class="ticks">${ticks}</div>`;
  }

  function cellAt(lng, lat) {
    const g = state.manifest.grid;
    const c = Math.floor((lng - g.lon0) / g.res), r = Math.floor((g.lat1 - lat) / g.res);
    if (c < 0 || r < 0 || c >= g.w || r >= g.h) return -1;
    return r * g.w + c;
  }

  function onMapMove(e) {
    const d = state.day; if (!d) return;
    const i = cellAt(e.lngLat.lng, e.lngLat.lat);
    const ro = $("readout");
    if (i < 0 || d.code[i] === 0) { ro.hidden = true; return; }
    const pm = d.pm[i], cat = S.aqiCategory(pm);
    const f = state.map.queryRenderedFeatures(e.point, { layers: ["adm2-fill", "adm1-fill"].filter(l => state.map.getLayer(l)) });
    const dist = f.find(x => x.layer.id === "adm2-fill"), reg = f.find(x => x.layer.id === "adm1-fill");
    const st = state.map.queryRenderedFeatures(e.point, { layers: ["stations"] })[0];
    ro.hidden = false;
    ro.innerHTML = `<div class="muted">${e.lngLat.lat.toFixed(3)}°N, ${e.lngLat.lng.toFixed(3)}°E${dist ? " · " + dist.properties.name : ""}${reg ? ", " + reg.properties.name : ""}</div>
      <div class="big">${C.fmt1(pm)} <span class="muted">µg/m³</span></div>
      <div class="row"><span>90% interval</span><span>${C.fmt1(d.lo[i])} – ${C.fmt1(d.hi[i])}</span></div>
      <div class="row"><span>AQI</span><span class="cat-chip"><i style="background:${cat.color}"></i>${S.aqiValue(pm)} · ${cat.label}</span></div>
      <div class="row"><span>CAMS input</span><span>${C.fmt1(d.cams[i])}</span></div>
      <div class="row"><span>Area of applicability</span><span>${d.code[i] === 2 ? "inside" : "outside (extrapolation)"}</span></div>
      ${st ? `<div class="row"><span>${st.properties.name}</span><span>${st.properties.has ? C.fmt1(st.properties.obs) + " obs." : "no obs."}</span></div>` : ""}`;
  }

  async function onMapClick(e) {
    const st = state.map.queryRenderedFeatures(e.point, { layers: ["stations"] })[0];
    state.selPoint = { lng: e.lngLat.lng, lat: e.lngLat.lat, station: st ? st.properties.site_id : null };
    await renderSeriesPanel();
  }

  function updateStationsLayer() {
    if (!state.mapReady) return;
    const show = $("chk-stations").checked;
    state.map.setLayoutProperty("stations", "visibility", show ? "visible" : "none");
    const obs = new Map((state.obsByDate.get(state.date) || []).map(o => [o.site_id, o]));
    const feats = state.stations.map(s => {
      const o = obs.get(s.site_id);
      const col = o ? S.colorAt(S.PM_STOPS, o.pm25) : (isDark() ? "#6f6d67" : "#b9b7ae");
      return { type: "Feature", geometry: { type: "Point", coordinates: [s.lon, s.lat] },
        properties: { site_id: s.site_id, name: s.name, ref: !!s.is_reference, has: !!o, obs: o ? o.pm25 : null, color: col } };
    });
    state.map.getSource("stations").setData({ type: "FeatureCollection", features: feats });
  }

  /* ============================== date & KPIs ============================== */
  function initControls() {
    const m = state.manifest;
    const di = $("date-input"); di.min = m.dates[0]; di.max = m.dates[m.dates.length - 1];
    di.onchange = () => setDate(nearestDate(di.value));
    $("d-prev").onclick = () => stepDate(-1);
    $("d-next").onclick = () => stepDate(1);
    $("d-latest").onclick = () => setDate(m.latest);
    $("d-play").onclick = togglePlay;
    $("layer-select").onchange = (e) => { state.layer = e.target.value; drawLayer(); };
    $("opacity").oninput = (e) => state.mapReady && state.map.setPaintProperty("pm", "raster-opacity", +e.target.value);
    $("chk-stations").onchange = updateStationsLayer;
    $("chk-aoa").onchange = drawLayer;
    $("adm-select").onchange = (e) => { for (const l of [1, 2]) if (state.map.getLayer(`adm${l}-line`)) state.map.setLayoutProperty(`adm${l}-line`, "visibility", String(l) === e.target.value ? "visible" : "none"); };
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (!$("tab-monitor").classList.contains("active")) return;
      if (e.key === "ArrowLeft") stepDate(-1);
      if (e.key === "ArrowRight") stepDate(1);
    });
    window.addEventListener("resize", debounce(() => rerenderAll(), 250));
  }
  const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
  function nearestDate(v) { const ds = state.manifest.dates; let best = ds[ds.length - 1]; for (const d of ds) if (d <= v) best = d; return best; }
  function stepDate(k) { const ds = state.manifest.dates; const i = ds.indexOf(state.date); const j = Math.max(0, Math.min(ds.length - 1, i + k)); if (j !== i) setDate(ds[j]); }
  function togglePlay() {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("d-play").textContent = "▶ Play"; return; }
    $("d-play").textContent = "❚❚ Pause";
    state.playing = setInterval(() => { const ds = state.manifest.dates; const i = ds.indexOf(state.date); if (i >= ds.length - 1) return togglePlay(); setDate(ds[i + 1]); }, 650);
  }

  async function getDay(date) {
    if (state.dayCache.has(date)) return state.dayCache.get(date);
    const p = S.loadDay(DATA, date);
    state.dayCache.set(date, p);
    if (state.dayCache.size > 40) state.dayCache.delete(state.dayCache.keys().next().value);
    return p;
  }

  async function setDate(date) {
    state.date = date;
    $("date-input").value = date;
    const tierName = (state.manifest.tiers || {})[date] || "final";
    $("tier-badge").textContent = tierName;
    $("tier-badge").title = { forecast: "Forecast: CAMS forecast + IFS meteorology, no observations", nowcast: "Nowcast: today's inputs, observations as received", nrt: "Near-real-time: re-processed daily as late data arrive", final: "Final: all inputs complete" }[tierName] || "";
    try {
      state.day = await getDay(date);
    } catch (e) {
      $("map-note").textContent = `Grid for ${date} unavailable`; return;
    }
    $("map-note").textContent = `GH-PM25 · ${date} · 0.01° grid`;
    drawLayer();
    renderKPIs();
    renderRegionTable();
    const i = state.manifest.dates.indexOf(date);
    [i + 1, i - 1].forEach(k => { const d = state.manifest.dates[k]; if (d) getDay(d).catch(() => {}); });
    if (state.selPoint) renderSeriesPanel(true);
  }

  function regionIndex() { return state.regions ? state.regions.dates.indexOf(state.date) : -1; }

  function renderKPIs() {
    const R = state.regions, t = regionIndex();
    const nobs = (state.manifest.n_obs || {})[state.date];
    $("k-obs").textContent = nobs != null ? nobs : "—";
    $("k-obs-sub").textContent = nobs ? "stations with valid daily mean (Stage-2 kriging)" : "no same-day observations (Stage 1 only)";
    if (!R || t < 0) { ["k-pw", "k-who", "k-gepa", "k-top"].forEach(k => $(k).textContent = "—"); return; }
    const a = R.adm0;
    $("k-pw").textContent = C.fmt1(a.pop_weighted[0][t]) + " µg/m³";
    $("k-pw-sub").textContent = `90% interval ${C.fmt1(a.lower90[0][t])}–${C.fmt1(a.upper90[0][t])} µg/m³`;
    $("k-who").textContent = C.fmtPct(a.pop_frac_gt15[0][t]);
    $("k-gepa").textContent = C.fmtPct(a.pop_frac_gt35[0][t]);
    const r1 = R.adm1; let best = -1, bv = -1;
    r1.names.forEach((n, k) => { const v = r1.pop_weighted[k][t]; if (v > bv) { bv = v; best = k; } });
    $("k-top").textContent = best >= 0 ? r1.names[best] : "—";
    $("k-top-sub").textContent = best >= 0 ? `${C.fmt1(bv)} µg/m³ population-weighted` : "";
  }

  function renderRegionTable() {
    const R = state.regions, t = regionIndex();
    if (!R || t < 0) return C.table("#region-table", [], []);
    const r1 = R.adm1;
    const rows = r1.names.map((n, k) => ({ k, name: n, pw: r1.pop_weighted[k][t], lo: r1.lower90[k][t], hi: r1.upper90[k][t], f35: r1.pop_frac_gt35[k][t] }));
    const vmax = d3.max(rows, r => r.pw) || 1;
    C.table("#region-table", [
      { key: "name", label: "Region" },
      { key: "pw", label: "PM2.5", num: true, fmt: (v) => `<span class="cat-chip"><i style="background:${S.colorAt(S.PM_STOPS, v)}"></i>${C.fmt1(v)}</span>` },
      { key: "lo", label: "90% interval", num: true, fmt: (v, r) => `${C.fmt1(r.lo)}–${C.fmt1(r.hi)}` },
      { key: "f35", label: "Pop. > 35", num: true, fmt: (v) => C.fmtPct(v) },
    ], rows, { sortKey: "pw", onRow: (r) => { selectExposureArea("adm1", r.k); document.querySelector('.tabs button[data-tab="exposure"]').click(); } });
  }

  /* ---------- location time series (cube) ---------- */
  async function loadCube(year) {
    if (state.cubes.has(year)) return state.cubes.get(year);
    const p = (async () => {
      const hdr = await fetchJSON(`cube/pm25_005_${year}.json`);
      const buf = await fetch(`${DATA}/cube/pm25_005_${year}.bin`).then(r => r.arrayBuffer());
      return { hdr, data: new Uint16Array(buf) };
    })();
    state.cubes.set(year, p);
    return p;
  }

  async function renderSeriesPanel(keepOnly) {
    const p = state.selPoint;
    if (!p) { C.timeSeries("#ts-chart", { series: [], emptyText: "Click anywhere on the map to plot the daily PM2.5 history of that location." }); return; }
    const years = [...new Set(state.manifest.dates.map(d => +d.slice(0, 4)))];
    const pts = [];
    for (const y of years) {
      let c; try { c = await loadCube(y); } catch (e) { continue; }
      const h = c.hdr;
      const col = Math.floor((p.lng - h.lon0) / h.res), row = Math.floor((h.lat0 - p.lat) / h.res);
      if (col < 0 || row < 0 || col >= h.nx || row >= h.ny) continue;
      h.dates.forEach((d, t) => { const v = c.data[t * h.ny * h.nx + row * h.nx + col]; pts.push({ date: new Date(d), value: v === h.nodata ? NaN : v * h.scale }); });
    }
    pts.sort((a, b) => a.date - b.date);
    const series = [{ name: "GH-PM25 (0.05° cell)", color: C.css("--series-1"), points: pts, width: 1.4 }];
    // nearest station within 10 km
    let near = null, nd = 1e9;
    state.stations.forEach(s => { const dk = Math.hypot((s.lon - p.lng) * 110.6 * Math.cos(p.lat * Math.PI / 180), (s.lat - p.lat) * 110.6); if (dk < nd) { nd = dk; near = s; } });
    if (p.station) near = state.stations.find(s => s.site_id === p.station) || near;
    if (near && (nd < 10 || p.station) && state.series[near.site_id]) {
      const s = state.series[near.site_id];
      series.push({ name: `Observed · ${near.name}`, color: C.css("--series-2"), dots: true, points: s.date.map((d, i) => ({ date: new Date(d), value: s.pm25[i] })) });
    }
    $("ts-title").textContent = p.station && near ? near.name : `${p.lat.toFixed(3)}°N, ${p.lng.toFixed(3)}°E`;
    $("ts-sub").textContent = series.length > 1 ? "model vs nearest monitor" : "daily mean, µg/m³";
    C.timeSeries("#ts-chart", { series, refLines: [{ y: 15, label: "WHO 15" }, { y: 35, label: "Ghana EPA 35" }], margin: { left: 36 } });
    const recent = pts.slice(-60);
    $("ts-table").innerHTML = `<table><thead><tr><th>Date</th><th class="num">PM2.5</th></tr></thead><tbody>${recent.reverse().map(q => `<tr><td>${q.date.toISOString().slice(0, 10)}</td><td class="num">${C.fmt1(q.value)}</td></tr>`).join("")}</tbody></table>`;
  }

  /* ============================== stations tab ============================== */
  function initStationsTab() {
    const nets = [...new Set(state.stations.map(s => s.network))].sort();
    $("st-network").innerHTML = `<option value="all">All networks</option>` + nets.map(n => `<option>${n}</option>`).join("");
    ["st-country", "st-network"].forEach(id => $(id).onchange = renderStationTable);
    $("st-search").oninput = debounce(renderStationTable, 150);
    renderStationTable();
  }
  function renderStationTable() {
    const cty = $("st-country").value, net = $("st-network").value, q = $("st-search").value.toLowerCase();
    const rows = state.stations.filter(s => (cty === "all" || s.country === cty) && (net === "all" || s.network === net) && (!q || s.name.toLowerCase().includes(q)))
      .map(s => { const se = state.series[s.site_id]; const last = se && se.date.length ? se.date[se.date.length - 1] : null; return { ...s, last, lastv: last ? se.pm25[se.pm25.length - 1] : null }; });
    $("st-count").textContent = `${rows.length} stations · ${rows.filter(r => r.is_reference).length} reference-grade`;
    C.table("#station-table", [
      { key: "name", label: "Site", fmt: (v, r) => `${v}${r.is_reference ? ' <span class="tier">BAM</span>' : ""}` },
      { key: "network", label: "Network" }, { key: "country", label: "Ctry" },
      { key: "n_days", label: "Days", num: true }, { key: "last", label: "Last obs." },
      { key: "lastv", label: "Last µg/m³", num: true, fmt: v => v == null ? "—" : C.fmt1(v) },
    ], rows, { sortKey: "n_days", onRow: (r) => { state.selStation = r; renderStation(r); } });
  }
  function renderStation(s) {
    const se = state.series[s.site_id];
    $("st-title").textContent = s.name;
    $("st-sub").textContent = `${s.network} · ${s.country} · ${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}${s.is_reference ? " · reference BAM-1020" : " · calibrated low-cost sensor"}`;
    if (!se) return C.timeSeries("#st-ts", { series: [] });
    const dates = se.date.map(d => new Date(d));
    const series = [
      { name: "Observed (calibrated)", color: C.css("--series-1"), points: dates.map((d, i) => ({ date: d, value: se.pm25[i] })), width: 1.6 },
      { name: "GH-PM25 cross-validated", color: C.css("--series-2"), points: dates.map((d, i) => ({ date: d, value: se.cv_pred ? se.cv_pred[i] : NaN })), width: 1.6 },
    ];
    if (!s.is_reference && se.pm25_raw) series.push({ name: "Raw sensor", color: C.css("--series-3"), points: dates.map((d, i) => ({ date: d, value: se.pm25_raw[i] })), width: 1 });
    C.timeSeries("#st-ts", { series, refLines: [{ y: 35, label: "Ghana EPA 35" }] });
    $("st-legend").innerHTML = series.map(x => `<span><i style="background:${x.color}"></i>${x.name}</span>`).join("");
    // scatter
    const pairs = se.pm25.map((o, i) => [o, se.cv_pred ? se.cv_pred[i] : null]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    scatterSimple("#st-scatter", pairs);
    const st = stats(pairs);
    $("st-stats").innerHTML = pairs.length ? `<table><tbody>
      <tr><td>Station-days</td><td class="num">${pairs.length}</td></tr><tr><td>Mean observed</td><td class="num">${C.fmt1(d3.mean(pairs, p => p[0]))} µg/m³</td></tr>
      <tr><td>R²</td><td class="num">${C.fmt2(st.r2)}</td></tr><tr><td>Pearson r</td><td class="num">${C.fmt2(st.r)}</td></tr><tr><td>RMSE</td><td class="num">${C.fmt1(st.rmse)} µg/m³</td></tr>
      <tr><td>MAE</td><td class="num">${C.fmt1(st.mae)} µg/m³</td></tr><tr><td>Mean bias</td><td class="num">${C.fmt1(st.mb)} µg/m³</td></tr></tbody></table>
      <p class="muted">Predictions are leave-city-cluster-out: no monitor within ~30 km of this site was used to train the model that predicted it.</p>` : `<div class="empty">No cross-validated predictions</div>`;
    $("st-table").innerHTML = `<table><thead><tr><th>Date</th><th class="num">Observed</th><th class="num">Raw</th><th class="num">CV prediction</th></tr></thead><tbody>` +
      se.date.slice().reverse().slice(0, 400).map((d, k) => { const i = se.date.length - 1 - k; return `<tr><td>${d}</td><td class="num">${C.fmt1(se.pm25[i])}</td><td class="num">${se.pm25_raw && se.pm25_raw[i] != null ? C.fmt1(se.pm25_raw[i]) : "—"}</td><td class="num">${se.cv_pred && se.cv_pred[i] != null ? C.fmt1(se.cv_pred[i]) : "—"}</td></tr>`; }).join("") + "</tbody></table>";
  }
  function stats(pairs) {
    if (pairs.length < 3) return {};
    const o = pairs.map(p => p[0]), m = pairs.map(p => p[1]);
    const mo = d3.mean(o), mm = d3.mean(m);
    const r = d3.sum(o.map((x, i) => (x - mo) * (m[i] - mm))) / Math.sqrt(d3.sum(o.map(x => (x - mo) ** 2)) * d3.sum(m.map(x => (x - mm) ** 2)));
    const r2 = 1 - d3.sum(o.map((x, i) => (m[i] - x) ** 2)) / d3.sum(o.map(x => (x - mo) ** 2));
    return { r, r2, rmse: Math.sqrt(d3.mean(o.map((x, i) => (m[i] - x) ** 2))), mae: d3.mean(o.map((x, i) => Math.abs(m[i] - x))), mb: mm - mo };
  }
  function scatterSimple(el, pairs) {
    const node = document.querySelector(el); node.innerHTML = "";
    if (!pairs.length) return;
    const W = node.clientWidth, H = node.clientHeight, m = { l: 40, r: 10, t: 8, b: 32 };
    const vmax = d3.max(pairs.flat()) * 1.05;
    const x = d3.scaleLinear().domain([0, vmax]).nice().range([m.l, W - m.r]), y = d3.scaleLinear().domain(x.domain()).range([H - m.b, m.t]);
    const svg = d3.select(node).append("svg").attr("width", W).attr("height", H);
    svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append("line").attr("class", "ref-line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(x.domain()[1])).attr("y2", y(x.domain()[1]));
    svg.append("g").selectAll("circle").data(pairs).join("circle").attr("cx", p => x(p[0])).attr("cy", p => y(p[1])).attr("r", 2.5).attr("fill", C.css("--series-1")).attr("opacity", 0.5);
    svg.append("text").attr("class", "ref-label").attr("x", (W + m.l) / 2).attr("y", H - 2).attr("text-anchor", "middle").text("Observed (µg/m³)");
  }

  /* ============================== exposure tab ============================== */
  function initExposureTab() {
    $("ex-level").onchange = () => { fillAreas(); renderExposure(); };
    $("ex-area").onchange = renderExposure;
    $("ex-smooth").onchange = renderExposure;
    fillAreas();
  }
  function fillAreas() {
    const R = state.regions; if (!R) return;
    const lvl = $("ex-level").value;
    const names = R[lvl].names.map((n, i) => ({ n, i })).sort((a, b) => a.n.localeCompare(b.n));
    $("ex-area").innerHTML = `<option value="nat">Ghana (national)</option>` + names.map(o => `<option value="${o.i}">${o.n}</option>`).join("");
  }
  function selectExposureArea(level, idx) { $("ex-level").value = level; fillAreas(); $("ex-area").value = String(idx); }
  function smooth(arr, k) {
    if (k <= 1) return arr;
    const out = new Array(arr.length).fill(NaN);
    for (let i = 0; i < arr.length; i++) { let s = 0, n = 0; for (let j = Math.max(0, i - k + 1); j <= i; j++) if (Number.isFinite(arr[j])) { s += arr[j]; n++; } out[i] = n >= Math.ceil(k / 2) ? s / n : NaN; }
    return out;
  }
  function areaSeries() {
    const R = state.regions, lvl = $("ex-level").value, a = $("ex-area").value;
    if (a === "nat") return { name: "Ghana", pw: R.adm0.pop_weighted[0], lo: R.adm0.lower90[0], hi: R.adm0.upper90[0] };
    const k = +a; return { name: R[lvl].names[k], pw: R[lvl].pop_weighted[k], lo: R[lvl].lower90[k], hi: R[lvl].upper90[k] };
  }
  function renderExposure() {
    const R = state.regions; if (!R) return;
    const k = +$("ex-smooth").value, A = areaSeries();
    const dates = R.dates.map(d => new Date(d));
    const nz = (v) => (v == null ? NaN : v);
    const pw = smooth(A.pw.map(nz), k), lo = smooth(A.lo.map(nz), k), hi = smooth(A.hi.map(nz), k);
    $("ex-title").textContent = `Population-weighted PM2.5 — ${A.name}`;
    C.timeSeries("#ex-ts", { series: [{ name: A.name, color: C.css("--series-1"), points: dates.map((d, i) => ({ date: d, value: pw[i] })) }],
      band: dates.map((d, i) => ({ date: d, lo: lo[i], hi: hi[i] })), refLines: [{ y: 15, label: "WHO 15" }, { y: 35, label: "Ghana EPA 35" }],
      onClick: (d) => { const s = d.toISOString().slice(0, 10); if (state.manifest.dates.includes(s)) { setDate(s); document.querySelector('.tabs button[data-tab="monitor"]').click(); } } });
    $("ex-table").innerHTML = monthlyTable(dates, A.pw.map(nz));
    renderSeason(dates, A.pw.map(nz));
    renderExceedance();
    renderNorthSouth(dates);
  }
  function monthlyTable(dates, v) {
    const g = d3.rollup(dates.map((d, i) => [d, v[i]]).filter(p => Number.isFinite(p[1])), a => d3.mean(a, p => p[1]), p => p[0].toISOString().slice(0, 7));
    return `<table><thead><tr><th>Month</th><th class="num">Mean PM2.5</th></tr></thead><tbody>${[...g].reverse().map(([m, x]) => `<tr><td>${m}</td><td class="num">${C.fmt1(x)}</td></tr>`).join("")}</tbody></table>`;
  }
  function renderSeason(dates, v) {
    const node = $("ex-season"); node.innerHTML = "";
    const byYear = d3.group(dates.map((d, i) => ({ y: d.getUTCFullYear(), m: d.getUTCMonth(), v: v[i] })).filter(p => Number.isFinite(p.v)), p => p.y);
    const years = [...byYear.keys()].sort();
    const colors = ["--series-1", "--series-2", "--series-3"].map(C.css).concat(["#eda100", "#e87ba4", "#008300"]);
    const W = node.clientWidth, H = node.clientHeight, m = { l: 40, r: 12, t: 10, b: 44 };
    const svg = d3.select(node).append("svg").attr("width", W).attr("height", H);
    const x = d3.scalePoint().domain(d3.range(12)).range([m.l, W - m.r]).padding(0.3);
    const lines = years.map(yr => ({ yr, pts: d3.range(12).map(mo => { const a = byYear.get(yr).filter(p => p.m === mo); return { m: mo, v: a.length >= 10 ? d3.mean(a, p => p.v) : NaN }; }) }));
    const ymax = d3.max(lines.flatMap(l => l.pts.map(p => p.v)).filter(Number.isFinite)) || 50;
    const y = d3.scaleLinear().domain([0, ymax * 1.1]).nice().range([H - m.b, m.t]);
    svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat(i => "JFMAMJJASOND"[i]));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    lines.forEach((l, i) => {
      svg.append("path").datum(l.pts).attr("fill", "none").attr("stroke", colors[i % colors.length]).attr("stroke-width", 2)
        .attr("d", d3.line().defined(p => Number.isFinite(p.v)).x(p => x(p.m)).y(p => y(p.v)));
      svg.append("g").selectAll("circle").data(l.pts.filter(p => Number.isFinite(p.v))).join("circle").attr("cx", p => x(p.m)).attr("cy", p => y(p.v)).attr("r", 4)
        .attr("fill", colors[i % colors.length]).attr("stroke", C.css("--surface-1")).attr("stroke-width", 2)
        .on("mousemove", (evt, p) => C.showTip(evt, `${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][p.m]} ${l.yr}`, [["mean PM2.5", C.fmt1(p.v) + " µg/m³"]])).on("mouseleave", C.hideTip);
    });
    const lg = svg.append("g").attr("transform", `translate(${m.l},${H - 14})`);
    lines.forEach((l, i) => { const gx = lg.append("g").attr("transform", `translate(${i * 64},0)`); gx.append("rect").attr("width", 14).attr("height", 3).attr("y", -4).attr("fill", colors[i % colors.length]); gx.append("text").attr("class", "ref-label").attr("x", 18).attr("y", 0).text(l.yr); });
  }
  function renderExceedance() {
    const R = state.regions, r1 = R.adm1;
    const years = [...new Set(R.dates.map(d => d.slice(0, 4)))];
    const rows = r1.names.map((n, k) => {
      const row = { name: n };
      years.forEach(y => { const idx = R.dates.map((d, i) => d.startsWith(y) ? i : -1).filter(i => i >= 0); const v = idx.map(i => r1.pop_weighted[k][i]).filter(x => x != null); row[y] = v.length >= 60 ? v.filter(x => x > 35).length / v.length : null; row["n" + y] = v.length; });
      return row;
    });
    const nDays = Object.fromEntries(years.map(y => [y, R.dates.filter(d => d.startsWith(y)).length]));
    C.table("#ex-exceed", [{ key: "name", label: "Region" }].concat(years.map(y => ({ key: y, label: nDays[y] < 360 ? `${y} (${nDays[y]} d)` : y, num: true, fmt: (v, r) => v == null ? `<span class="muted">n=${r["n" + y]}</span>` : C.fmtPct(v) }))), rows, { sortKey: years[years.length - 1] });
  }
  function renderNorthSouth(dates) {
    const R = state.regions, r1 = R.adm1;
    const pop = r1.population || r1.names.map(() => 1);
    const grp = (north) => dates.map((d, t) => { let s = 0, w = 0; r1.names.forEach((n, k) => { if (NORTH.includes(n) === north) { const v = r1.pop_weighted[k][t]; if (v != null) { s += v * pop[k]; w += pop[k]; } } }); return w ? s / w : NaN; });
    const k = 30;
    C.timeSeries("#ex-ns", { series: [
      { name: "Northern Ghana (30-day mean)", color: C.css("--series-2"), points: smooth(grp(true), k).map((v, i) => ({ date: dates[i], value: v })) },
      { name: "Southern & middle belt (30-day mean)", color: C.css("--series-1"), points: smooth(grp(false), k).map((v, i) => ({ date: dates[i], value: v })) },
    ], refLines: [{ y: 35, label: "Ghana EPA 35" }] });
  }

  /* ============================== validation tab ============================== */
  async function initValidationTab() {
    try { state.validation = await fetchJSON("validation/validation.json"); } catch (e) { state.validation = null; }
    $("cv-scheme").onchange = renderValidation; $("cv-subset").onchange = renderValidation;
  }
  const MODEL_LABEL = { cams_raw: "CAMS raw", cams_linear: "CAMS + linear bias correction", paperlike_gbdt: "GBDT, AOD + met + DOY (baseline-style)", lgbm: "M1 LightGBM", xgb_ratio: "M2 XGBoost (CAMS ratio)", catboost: "M3 CatBoost", extratrees: "M4 ExtraTrees", stack: "GH-PM25 ensemble (stage 1)", stack_rk: "GH-PM25 final (operational)" };
  function renderValidation() {
    const V = state.validation; if (!V) return;
    const sch = $("cv-scheme").value, sub = $("cv-subset").value;
    const rows = V.metrics.filter(r => r.scheme === sch && r.subset === sub);
    const fin = rows.find(r => r.model === "stack_rk"), cams = rows.find(r => r.model === "cams_raw");
    $("cv-kpis").innerHTML = fin ? [
      ["R² (final model)", C.fmt2(fin.r2), `CAMS raw: ${cams ? C.fmt2(cams.r2) : "—"}`],
      ["RMSE", C.fmt1(fin.rmse) + " µg/m³", `CAMS raw: ${cams ? C.fmt1(cams.rmse) : "—"}`],
      ["MAE", C.fmt1(fin.mae) + " µg/m³", `mean observed ${C.fmt1(fin.mean_obs)}`],
      ["Mean bias", C.fmt1(fin.mb) + " µg/m³", `NMB ${C.fmtPct(fin.nmb)}`],
      ["Station-days", d3.format(",")(fin.n), `regression slope ${C.fmt2(fin.slope)}`],
    ].map(([l, v, s]) => `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value">${v}</div><div class="kpi-sub">${s}</div></div>`).join("") : "";
    const order = ["cams_raw", "cams_linear", "paperlike_gbdt", "lgbm", "xgb_ratio", "catboost", "extratrees", "stack", "stack_rk"];
    const br = order.map(k => rows.find(r => r.model === k)).filter(Boolean).map(r => ({
      label: MODEL_LABEL[r.model], value: r.r2, direct: true,
      color: r.model.startsWith("stack") ? C.css("--series-1") : r.model.startsWith("cams") || r.model === "paperlike_gbdt" ? (isDark() ? "#6f6d67" : "#b9b7ae") : C.css("--accent-soft"),
      tip: [["R²", C.fmt3(r.r2)], ["RMSE", C.fmt1(r.rmse) + " µg/m³"], ["MAE", C.fmt1(r.mae) + " µg/m³"], ["bias", C.fmt1(r.mb)], ["n", d3.format(",")(r.n)]],
    }));
    C.hbars("#cv-bars", br, { labelWidth: 250, max: 1, min: Math.min(0, d3.min(br, b => b.value)), valueFormat: C.fmt2 });
    C.table("#cv-table", [
      { key: "model", label: "Model", fmt: v => MODEL_LABEL[v] || v }, { key: "r2", label: "R²", num: true, fmt: C.fmt3 }, { key: "r", label: "r", num: true, fmt: C.fmt3 },
      { key: "rmse", label: "RMSE", num: true, fmt: C.fmt1 }, { key: "mae", label: "MAE", num: true, fmt: C.fmt1 }, { key: "mb", label: "Bias", num: true, fmt: C.fmt1 },
      { key: "slope", label: "Slope", num: true, fmt: C.fmt2 }, { key: "n", label: "n", num: true },
    ], order.map(k => rows.find(r => r.model === k)).filter(Boolean), { rowClass: r => r.model === "stack_rk" ? "highlight" : "" });
    C.densityScatter("#cv-scatter", (V.scatter || {})[sch]);
    const sh = (V.shap || []).slice(0, 20).map(r => ({ label: r.feature, value: r.mean_abs_shap, color: C.css("--series-1") }));
    C.hbars("#shap-bars", sh, { labelWidth: 140, valueFormat: C.fmt3 });
    C.correlogram("#corr-chart", V.correlogram);
    C.table("#coverage-table", [{ key: "subset", label: "Subset" }, { key: "nominal", label: "Nominal", num: true, fmt: C.fmtPct }, { key: "coverage", label: "Empirical", num: true, fmt: v => C.fmtPct(v) }, { key: "median_width", label: "Median width", num: true, fmt: C.fmt1 }], V.coverage || []);
    C.table("#cal-table", [{ key: "key", label: "Model" }, { key: "n_pairs", label: "Pairs", num: true }, { key: "n_sensors", label: "Sensors", num: true },
      { key: "raw_r2", label: "R² raw", num: true, fmt: C.fmt2 }, { key: "cv_r2", label: "R² cal.", num: true, fmt: C.fmt2 },
      { key: "raw_mae", label: "MAE raw", num: true, fmt: C.fmt1 }, { key: "cv_mae", label: "MAE cal.", num: true, fmt: C.fmt1 },
      { key: "raw_bias", label: "Bias raw", num: true, fmt: C.fmt1 }, { key: "cv_bias", label: "Bias cal.", num: true, fmt: C.fmt1 }], V.calibration || []);
    C.table("#abl-table", [{ key: "variant", label: "Variant" }, { key: "subset", label: "Subset" }, { key: "n_features", label: "Feat.", num: true }, { key: "r2", label: "R²", num: true, fmt: C.fmt3 }, { key: "rmse", label: "RMSE", num: true, fmt: C.fmt1 }], V.ablation || []);
    C.table("#site-metrics", [{ key: "name", label: "Site" }, { key: "network", label: "Network" }, { key: "country", label: "Ctry" }, { key: "n", label: "Days", num: true },
      { key: "mean_obs", label: "Mean obs.", num: true, fmt: C.fmt1 }, { key: "r2", label: "R²", num: true, fmt: C.fmt2 }, { key: "r", label: "r", num: true, fmt: C.fmt2 },
      { key: "rmse", label: "RMSE", num: true, fmt: C.fmt1 }, { key: "mb", label: "Bias", num: true, fmt: C.fmt1 }], V.per_site || [], { sortKey: "n" });
  }

  /* ============================== methods ============================== */
  function renderMethods() {
    const m = state.manifest, hl = m.headline || {};
    const src = (m.sources || []).map(s => `<tr><td>${s.name}</td><td>${s.role}</td><td>${s.resolution}</td><td>${s.access}</td></tr>`).join("");
    $("methods-body").innerHTML = `
      <h2>About GH-PM25</h2>
      <p>GH-PM25 estimates daily mean surface PM2.5 on a 0.01° (~1.1 km) grid across Ghana. It fuses ground monitors with satellite and model data:</p>
      <ol><li>An ensemble of gradient-boosted and randomised tree learners, trained on ${m.training ? d3.format(",")(m.training.n_train) : "—"} calibrated station-days from ${m.training ? m.training.n_sites : "—"} monitors in Ghana and neighbouring countries. Its hyperparameters and composition were chosen under spatial cross-validation.</li>
      <li>A same-day residual-kriging stage. Its strength is set by leave-site-out validation, and it is currently switched off because it did not add skill with today's sensor networks.</li>
      <li>Distribution-free, locally adaptive conformal prediction intervals, plus an area-of-applicability mask.</li></ol>
      <div class="diagram">${pipelineSVG()}</div>
      <h3>Product tiers</h3>
      <ul><li><b>Forecast (D+1)</b> — CAMS forecast and bias-adjusted ECMWF IFS meteorology; no observations.</li>
      <li><b>Nowcast (D0)</b> — same-day inputs, with any observations already received.</li>
      <li><b>Near-real-time (D−1…D−6)</b> — re-run each day as late inputs arrive (MERRA-2 via NASA POWER, OpenAQ archive).</li>
      <li><b>Final (≤ D−7)</b> — all inputs complete.</li></ul>
      <h3>Input data</h3>
      <div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Role</th><th>Resolution</th><th>Access</th></tr></thead><tbody>${src}</tbody></table></div>
      <h3>Headline accuracy (leave-city-cluster-out cross-validation)</h3>
      <p>${hl.text || "See the Model &amp; validation tab."}</p>
      <h3>Interpreting the maps</h3>
      <ul><li>Values are 24-h means (00–24 UTC, the same as Ghana local time) in µg/m³, referenced to BAM-1020 measurements.</li>
      <li>The 90% interval is expected to contain the true daily mean in 9 of 10 cases, based on held-out stations in cities the model never saw.</li>
      <li>Cells outside the area of applicability have predictor combinations unlike anything in training (for example, far-north Harmattan dust extremes); treat them as extrapolation.</li>
      <li>Northern Ghana has few monitors, so uncertainty there is wider. The intervals reflect this.</li></ul>
      <h3>Downloads &amp; citation</h3>
      <p>Daily NetCDF (CF-1.8), monthly and annual GeoTIFF, and regional CSV statistics are in <code>outputs/</code>. The full methodology, data provenance and evaluation are in the <a href="report/GH-PM25_Technical_Report.html">technical report</a>.</p>`;
  }
  function pipelineSVG() {
    const ink = "var(--text-secondary)", box = "var(--surface-2)", line = "var(--axis)";
    const b = (x, y, w, h, t, s) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${box}" stroke="${line}"/><text x="${x + w / 2}" y="${y + 20}" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--text-primary)">${t}</text>${s.map((l, i) => `<text x="${x + w / 2}" y="${y + 38 + i * 15}" text-anchor="middle" font-size="11" fill="${ink}">${l}</text>`).join("")}</g>`;
    const a = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${line}" stroke-width="1.5" marker-end="url(#arr)"/>`;
    return `<svg viewBox="0 0 980 300" width="100%" style="min-width:760px" role="img" aria-label="GH-PM25 processing pipeline">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${line}"/></marker></defs>
      ${b(10, 10, 200, 80, "Ground monitors", ["OpenAQ · AirNow embassies", "QC → daily means", "LCS calibration vs BAM"])}
      ${b(10, 110, 200, 80, "Atmosphere & weather", ["CAMS composition (0.4°)", "MERRA-2 met + AOD (0.5°)", "VIIRS active fires (375 m)"])}
      ${b(10, 210, 200, 80, "Static 1-km context", ["WorldCover · Copernicus DEM", "WorldPop · VIIRS night lights", "OSM roads · coast distance"])}
      ${b(270, 110, 180, 80, "Feature engineering", ["bilinear to 1 km · lags", "ventilation · dust fraction", "upwind fire influence"])}
      ${b(500, 20, 200, 80, "Stage 1: tree ensemble", ["4 learners tuned under spatial CV", "selected: CatBoost + ExtraTrees", "equal-weight log-space average"])}
      ${b(500, 120, 200, 70, "Stage 2: residual kriging", ["site-offset-removed anomalies", "strength λ set by site-out CV"])}
      ${b(500, 210, 200, 80, "Stage 3: uncertainty", ["normalised split-conformal", "90% intervals", "area of applicability"])}
      ${b(760, 110, 210, 80, "Products", ["daily 1-km NetCDF / PNG", "regional exposure statistics", "portal · forecasts · reports"])}
      ${a(210, 50, 270, 140)}${a(210, 150, 270, 150)}${a(210, 250, 270, 160)}
      ${a(450, 150, 500, 60)}${a(600, 100, 600, 120)}${a(600, 190, 600, 210)}${a(700, 155, 760, 150)}${a(700, 250, 760, 165)}
      ${a(110, 90, 110, 90)}</svg>`;
  }

  init();
})();
