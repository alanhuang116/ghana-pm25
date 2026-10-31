/* GH-PM25 Observatory - Greater Accra view of the Monitor map.
   Crops the 1-km grid to Greater Accra Region (app.js stretches the colours to the range inside it), and adds
   district statistics, TAHMO weather-station values on the map, local weather and the TAHMO experiment results. */
(function () {
  "use strict";
  const S = window.GHScales, C = window.GHCharts, A = window.GHApp;
  const { state, $ } = A;
  const REGION = "Greater Accra Region";
  const VIEW = [[-0.52, 5.47], [0.72, 6.10]];   // Greater Accra extent
  const WX = { rh: ["Relative humidity", "%"], t2m: ["Air temperature", "°C"], precip: ["Rainfall", "mm/day"], ws10: ["Wind speed (≈2 m)", "m/s"] };
  const PIN = { t2m: v => `${C.fmt1(v)}°`, rh: v => `${Math.round(v)}%`, precip: v => `${C.fmt1(v)} mm`, ws10: v => `${C.fmt1(v)} m/s` };
  // one-hue sequential ramp from the portal palette; dark mode runs dark -> light on the dark surface
  const BLUES_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const BLUES_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"];
  const acc = { districts: [], adm1: -1, tahmo: null, mask: null, pins: [], panels: false, layers: false, active: false };

  A.on((what) => {
    if (what === "ready") { acc.prepared = prepare(); initControls(); }
    if (what === "scope" || what === "map") state.scope === "accra" ? enter() : leave();
    if (!acc.active) return;
    if (what === "date") { renderKPIs(); renderDistricts(); renderPins(); }
    if (what === "theme") { setOutsideColour(); renderPins(); renderWeather(); renderTrend(); }
    if (what === "tab:monitor") setTimeout(() => { renderWeather(); renderTrend(); }, 60);
  });

  /* ---------- data and the Greater Accra cell mask ---------- */
  async function prepare() {
    const R = state.regions;
    if (R) acc.adm1 = R.adm1.names.indexOf(REGION);
    const [adm1, adm2, tahmo] = await Promise.all([
      A.fetchJSON("boundaries/adm1.geojson").catch(() => null), A.fetchJSON("boundaries/adm2.geojson").catch(() => null),
      A.fetchJSON("accra/tahmo.json").catch(() => null)]);
    acc.tahmo = tahmo;
    const ga = adm1 && adm1.features.find(f => f.properties.name === REGION);
    acc.gaFeature = ga;
    if (!ga) return;
    acc.mask = rasterMask(ga.geometry, state.manifest.grid);
    if (adm2 && R) {
      const inGA = adm2.features.filter(f => d3.geoContains(ga, d3.geoCentroid(f)));
      acc.adm2 = { type: "FeatureCollection", features: inGA };
      acc.districts = inGA.map(f => ({ name: f.properties.name, k: R.adm2.names.indexOf(f.properties.name) })).filter(d => d.k >= 0);
    }
  }

  // 1 for grid cells whose centre lies inside the polygon (scanline, even-odd over all rings)
  function rasterMask(geom, g) {
    const m = new Uint8Array(g.w * g.h);
    const rings = (geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates).flat();
    const lats = rings.flat().map(p => p[1]);
    const r0 = Math.max(0, Math.floor((g.lat1 - d3.max(lats)) / g.res)), r1 = Math.min(g.h - 1, Math.ceil((g.lat1 - d3.min(lats)) / g.res));
    for (let r = r0; r <= r1; r++) {
      const y = g.lat1 - (r + 0.5) * g.res, xs = [];
      for (const ring of rings) for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > y) !== (yj > y)) xs.push(xi + (y - yi) * (xj - xi) / (yj - yi));
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const c0 = Math.max(0, Math.ceil((xs[k] - g.lon0) / g.res - 0.5)), c1 = Math.min(g.w - 1, Math.floor((xs[k + 1] - g.lon0) / g.res - 0.5));
        for (let c = c0; c <= c1; c++) m[r * g.w + c] = 1;
      }
    }
    return m;
  }

  function initControls() {
    $("wx-map-var").onchange = () => {
      const v = $("wx-map-var").value;
      if (v) { $("a-wx-var").value = v; renderWeather(); }
      renderPins();
    };
    $("a-wx-var").value = $("wx-map-var").value || "t2m";
  }

  /* ---------- entering and leaving the Greater Accra view ---------- */
  async function enter() {
    await acc.prepared;
    if (state.scope !== "accra" || !state.mapReady || !acc.mask) return;
    const map = state.map, first = !acc.active;
    acc.active = true;
    state.cellMask = acc.mask;
    addLayers(); setOutsideColour();   // the theme may have changed while the Ghana view was open
    for (const l of ["ga-outside", "ga-line"]) map.setLayoutProperty(l, "visibility", "visible");
    for (const l of [1, 2]) if (map.getLayer(`adm${l}-line`)) map.setLayoutProperty(`adm${l}-line`, "visibility", l === 2 ? "visible" : "none");
    if (first) fitView();
    A.drawLayer();
    if (!acc.panels) { acc.panels = true; initWeather(); renderExperiments(); }
    renderKPIs(); renderDistricts(); renderPins();
    setTimeout(() => { renderWeather(); renderTrend(); }, 60);
  }
  // fit Greater Accra above the legends in the lower-left corner
  function fitView() {
    renderPins();
    const h = document.querySelector(".legend-stack").offsetHeight, H = state.map.getContainer().clientHeight;
    state.map.fitBounds(VIEW, { padding: { top: 20, right: 20, left: 20, bottom: Math.min(h + 16, H * 0.45) }, duration: 600 });
  }
  function leave() {
    if (!acc.active) return;
    acc.active = false;
    state.cellMask = null;
    const map = state.map;
    for (const l of ["ga-outside", "ga-line"]) if (map.getLayer(l)) map.setLayoutProperty(l, "visibility", "none");
    const lvl = $("adm-select").value;
    for (const l of [1, 2]) if (map.getLayer(`adm${l}-line`)) map.setLayoutProperty(`adm${l}-line`, "visibility", String(l) === lvl ? "visible" : "none");
    clearPins();
    $("wx-legend").hidden = true;
  }

  function addLayers() {
    if (acc.layers) return;
    acc.layers = true;
    const map = state.map, ga = acc.gaFeature.geometry;
    // dim everything outside Greater Accra: a frame around Ghana with the region cut out
    const holes = (ga.type === "Polygon" ? [ga.coordinates] : ga.coordinates).map(p => p[0]);
    map.addSource("ga-outside", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Polygon",
      coordinates: [[[-6, 2], [4, 2], [4, 14], [-6, 14], [-6, 2]], ...holes] } } });
    map.addLayer({ id: "ga-outside", type: "fill", source: "ga-outside", paint: { "fill-color": C.css("--page"), "fill-opacity": 0.6 } }, "labels");
    map.addSource("ga-line", { type: "geojson", data: acc.gaFeature });
    map.addLayer({ id: "ga-line", type: "line", source: "ga-line", paint: { "line-color": A.isDark() ? "#d9d8d2" : "#3d3c39", "line-width": 1.8 } }, "stations");
  }
  function setOutsideColour() {
    if (!acc.layers) return;
    state.map.setPaintProperty("ga-outside", "fill-color", C.css("--page"));
    state.map.setPaintProperty("ga-line", "line-color", A.isDark() ? "#d9d8d2" : "#3d3c39");
  }

  /* ---------- weather-station values on the map ---------- */
  const shiftDay = (d, k) => new Date(Date.parse(d) + k * 864e5).toISOString().slice(0, 10);
  function valueAt(code, v, d) {
    const s = acc.tahmo.series[code]; if (!s || !s[v]) return null;
    const k = s.date.indexOf(d);
    return k >= 0 && s[v][k] != null ? s[v][k] : null;
  }
  // the map date, or an earlier day (up to 3 back) when more stations have values then: today's daily mean is
  // usually still incomplete when the portal updates
  function stationDay(v) {
    let best = null, nb = 0;
    for (let k = 0; k <= 3; k++) {
      const d = shiftDay(state.date, -k), n = acc.tahmo.stations.filter(s => valueAt(s.code, v, d) != null).length;
      if (n > nb) { best = d; nb = n; }
    }
    return best;
  }
  // stations that reported in the 30 days before `day` (the others have stopped and get no pin)
  function reporting(code, day) {
    const s = acc.tahmo.series[code]; if (!s || !day) return false;
    const from = shiftDay(day, -30);
    return s.date.some(d => d > from && d <= day);
  }
  function clearPins() { acc.pins.forEach(m => m.remove()); acc.pins = []; }
  function renderPins() {
    clearPins();
    const T = acc.tahmo, v = $("wx-map-var").value, lg = $("wx-legend");
    if (!acc.active || !v || !T || !T.stations.length) { lg.hidden = true; return; }
    const day = stationDay(v);
    const vals = T.stations.filter(s => reporting(s.code, day)).map(s => ({ s, x: valueAt(s.code, v, day) }));
    const xs = vals.map(o => o.x).filter(x => x != null);
    let lo = v === "precip" ? 0 : d3.min(xs), hi = d3.max(xs);
    if (xs.length && hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const ramp = A.isDark() ? BLUES_DARK : BLUES_LIGHT, col = d3.scaleSequential(d3.piecewise(d3.interpolateRgb, ramp)).domain([lo, hi]);
    const sel = $("a-wx-station").value;
    for (const { s, x } of vals) {
      const el = document.createElement("div");
      el.className = `wx-pin${x == null ? " none" : ""}${s.code === sel ? " sel" : ""}`;
      const wd = v === "ws10" && day ? valueAt(s.code, "wd", day) : null;
      const arrow = wd != null ? `<svg viewBox="0 0 14 14" style="width:14px;height:14px;transform:rotate(${Math.round(wd + 180) % 360}deg)" aria-hidden="true"><path d="M7 12.5V2M3 6l4-4 4 4" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>` : "";
      el.innerHTML = `<span class="sw" style="${x == null ? "" : `background:${col(x)}`}"></span>${x == null ? "–" : PIN[v](x)}${arrow}`;
      el.title = `${s.name} (${s.code}) · ${WX[v][0]}: ${x == null ? "no data" : `${C.fmt1(x)} ${WX[v][1]}`}${day ? ` on ${day}` : ""}${wd != null ? ` · wind from ${Math.round(wd)}°` : ""}`;
      el.onclick = (e) => { e.stopPropagation(); $("a-wx-station").value = s.code; renderWeather(); renderPins(); };
      acc.pins.push(new maplibregl.Marker({ element: el }).setLngLat([s.lon, s.lat]).addTo(state.map));
    }
    lg.hidden = false;
    if (!xs.length) {
      lg.innerHTML = `<div class="title">${WX[v][0]} at weather stations</div><div class="sub">No station data within 3 days of ${state.date}.</div>`;
      return;
    }
    const f = v === "rh" ? d3.format(".0f") : d3.format(".1f");
    lg.innerHTML = `<div class="title">${WX[v][0]} at weather stations (${WX[v][1]})</div>
      <div class="ramp" style="width:180px;background:linear-gradient(90deg,${ramp.join(",")})"></div>
      <div class="ticks" style="width:180px"><span style="left:0;transform:none">${f(lo)}</span><span style="left:100%;transform:translateX(-100%)">${f(hi)}</span></div>
      <div class="sub">TAHMO daily means for ${day}${day !== state.date ? " (latest complete day)" : ""}${v === "ws10" ? "; arrows point where the wind blows" : ""}.</div>`;
  }

  /* ---------- KPIs and district table ---------- */
  function tIndex() { return state.regions ? state.regions.dates.indexOf(state.date) : -1; }
  function renderKPIs() {
    const R = state.regions, t = tIndex(), k = acc.adm1, d = state.day, m = acc.mask;
    if (d && m) {
      const v = [];
      for (let i = 0; i < m.length; i++) if (m[i] && d.code[i] !== 0 && Number.isFinite(d.pm[i])) v.push(d.pm[i]);
      v.sort((a, b) => a - b);
      $("a-k-range").textContent = v.length ? `${C.fmt1(d3.quantileSorted(v, 0.05))}–${C.fmt1(d3.quantileSorted(v, 0.95))}` : "—";
      $("a-k-range-sub").textContent = v.length ? `µg/m³ across ${d3.format(",")(v.length)} 1-km cells (5th–95th percentile)` : "";
    }
    if (!R || t < 0 || k < 0) { ["a-k-pw", "a-k-who", "a-k-gepa", "a-k-top"].forEach(id => $(id).textContent = "—"); return; }
    const a = R.adm1;
    $("a-k-pw").textContent = C.fmt1(a.pop_weighted[k][t]) + " µg/m³";
    $("a-k-pw-sub").textContent = `90% interval ${C.fmt1(a.lower90[k][t])}–${C.fmt1(a.upper90[k][t])} µg/m³`;
    $("a-k-who").textContent = C.fmtPct(a.pop_frac_gt15[k][t]);
    $("a-k-gepa").textContent = C.fmtPct(a.pop_frac_gt35[k][t]);
    let best = null;
    acc.districts.forEach(x => { const v = R.adm2.pop_weighted[x.k][t]; if (v != null && (!best || v > best.v)) best = { name: x.name, v }; });
    $("a-k-top").textContent = best ? best.name : "—";
    $("a-k-top-sub").textContent = best ? `${C.fmt1(best.v)} µg/m³ population-weighted` : "";
  }
  function renderDistricts() {
    const R = state.regions, t = tIndex();
    if (!R || t < 0 || !acc.districts.length) return C.table("#a-district-table", [], []);
    const a2 = R.adm2;
    const rows = acc.districts.map(d => ({ ...d, pw: a2.pop_weighted[d.k][t], lo: a2.lower90[d.k][t], hi: a2.upper90[d.k][t], f35: a2.pop_frac_gt35[d.k][t] }));
    C.table("#a-district-table", [
      { key: "name", label: "District" },
      { key: "pw", label: "PM2.5", num: true, fmt: (v) => v == null ? "—" : `<span class="cat-chip"><i style="background:${S.colorAt(S.PM_STOPS, v)}"></i>${C.fmt1(v)}</span>` },
      { key: "lo", label: "90% interval", num: true, fmt: (v, r) => r.lo == null ? "—" : `${C.fmt1(r.lo)}–${C.fmt1(r.hi)}` },
      { key: "f35", label: "Pop. > 35", num: true, fmt: (v) => v == null ? "—" : C.fmtPct(v) },
    ], rows, { sortKey: "pw", onRow: (r) => zoomTo(r.name) });
  }
  function zoomTo(name) {
    if (!state.mapReady || !acc.adm2) return;
    const f = acc.adm2.features.find(x => x.properties.name === name); if (!f) return;
    state.map.fitBounds(d3.geoBounds(f), { padding: 40, maxZoom: 12 });
  }

  /* ---------- trend: Greater Accra vs Ghana ---------- */
  function renderTrend() {
    const R = state.regions; if (!acc.active || !R || acc.adm1 < 0 || !$("a-trend").clientWidth) return;
    const dates = R.dates.map(d => new Date(d)), nz = v => (v == null ? NaN : v);
    const ga = A.smooth(R.adm1.pop_weighted[acc.adm1].map(nz), 7), gh = A.smooth(R.adm0.pop_weighted[0].map(nz), 7);
    const series = [
      { name: "Greater Accra", color: C.css("--series-1"), points: dates.map((d, i) => ({ date: d, value: ga[i] })), width: 1.8 },
      { name: "Ghana", color: C.css("--series-2"), points: dates.map((d, i) => ({ date: d, value: gh[i] })), width: 1.4 },
    ];
    C.timeSeries("#a-trend", { series, refLines: [{ y: 15, label: "WHO 15" }, { y: 35, label: "Ghana EPA 35" }] });
    $("a-trend-legend").innerHTML = series.map(x => `<span><i style="background:${x.color}"></i>${x.name}</span>`).join("");
  }

  /* ---------- TAHMO weather ---------- */
  function initWeather() {
    const T = acc.tahmo;
    if (!T || !T.stations.length) {
      $("a-wx-station").disabled = true; $("a-wx-var").disabled = true; $("wx-map-var").disabled = true;
      $("a-wx-chart").innerHTML = `<div class="empty">No weather-station data published yet. It appears here after the TAHMO download runs with the project's credentials.</div>`;
      return;
    }
    // stations that stopped reporting are labelled; the default is the most recent one nearest the city centre
    const last = s => { const d = (T.series[s.code] || {}).date || []; return d.length ? d[d.length - 1] : ""; };
    const latest = d3.max(T.stations, last), old = s => (new Date(latest) - new Date(last(s) || 0)) / 864e5 > 30;
    $("a-wx-station").innerHTML = T.stations.map(s => `<option value="${s.code}">${s.name} (${s.code})${old(s) ? ` · until ${last(s) || "n/a"}` : ""}</option>`).join("");
    const dist = s => Math.hypot(s.lat - 5.60, (s.lon + 0.19) * Math.cos(5.6 * Math.PI / 180));  // to the centre used in scripts/12_tahmo_accra.py
    $("a-wx-station").value = T.stations.filter(s => !old(s)).sort((a, b) => dist(a) - dist(b)).concat(T.stations)[0].code;
    $("a-wx-station").onchange = () => { renderWeather(); renderPins(); };
    $("a-wx-var").onchange = renderWeather;
    $("a-wx-sub").textContent = `${T.stations.length} TAHMO station${T.stations.length > 1 ? "s" : ""} · daily, updated ${T.generated_at.slice(0, 10)}`;
  }
  function renderWeather() {
    const T = acc.tahmo; if (!acc.active || !T || !T.stations.length || !$("a-wx-chart").clientWidth) return;
    const code = $("a-wx-station").value, v = $("a-wx-var").value, s = T.series[code];
    if (!s || !s[v]) return C.timeSeries("#a-wx-chart", { series: [], emptyText: "No data for this variable at this station" });
    const pts = s.date.map((d, i) => ({ date: new Date(d), value: s[v][i] == null ? NaN : s[v][i] }));
    C.timeSeries("#a-wx-chart", { series: [{ name: WX[v][0], color: C.css("--series-3"), points: pts, width: 1.4, dots: v === "precip" }], yLabel: WX[v][1] });
  }

  /* ---------- experiment results ---------- */
  function renderExperiments() {
    const E = acc.tahmo && acc.tahmo.experiments, el = $("a-exp");
    if (!E) {
      el.innerHTML = `<p class="note">Three experiments are ready to run once weather-station data is in: how far the model's weather input (MERRA-2, about 50 km) is from the ground in Accra; whether measured humidity improves the low-cost sensor calibration against the US Embassy monitor; and whether local weather improves the PM2.5 model for Accra. Results will be listed here.</p>`;
      return;
    }
    $("a-exp-sub").textContent = `${E.n_stations} stations, ${d3.format(",")(E.n_station_days)} station-days · run ${E.generated_at.slice(0, 10)}`;
    el.innerHTML = `<div class="exp-grid"><div><h4>Model weather input vs ground</h4><div id="a-exp-wx" class="table-wrap"></div></div>
      <div><h4>Sensor calibration: humidity source</h4><div id="a-exp-cal" class="table-wrap"></div></div>
      <div><h4>PM2.5 model in Accra</h4><div id="a-exp-model" class="table-wrap"></div></div></div>
      <p class="note">Errors are cross-validated. Full tables: <code>report/tahmo/TAHMO_Accra_Experiments.md</code> in the repository.</p>`;
    const lab = { t2m: "Temperature (°C)", t2m_range: "Daily range (°C)", rh: "Humidity (%)", precip: "Rain (mm)", ws10: "Wind (m/s)", ps: "Pressure (kPa)", ssrd: "Radiation (kWh/m² per day)" };
    C.table("#a-exp-wx", [{ key: "variable", label: "Variable", fmt: v => lab[v] || v }, { key: "merra_bias", label: "Bias", num: true, fmt: C.fmt2 },
      { key: "merra_rmse", label: "RMSE", num: true, fmt: C.fmt2 }, { key: "merra_r", label: "r", num: true, fmt: C.fmt2 }, { key: "n", label: "Days", num: true }], E.weather_bias || []);
    C.table("#a-exp-cal", [{ key: "network", label: "Sensors" }, { key: "humidity", label: "Humidity" }, { key: "correction", label: "Correction" },
      { key: "mae", label: "MAE", num: true, fmt: C.fmt2 }, { key: "r2", label: "R²", num: true, fmt: C.fmt2 }, { key: "n", label: "Pairs", num: true }],
      (E.calibration || []).filter(r => r.correction !== "ratio"));
    C.table("#a-exp-model", [{ key: "design", label: "Design" }, { key: "weather", label: "Weather input" }, { key: "rmse", label: "RMSE", num: true, fmt: C.fmt2 },
      { key: "r2", label: "R²", num: true, fmt: C.fmt2 }, { key: "n", label: "Days", num: true }], E.model || []);
  }
})();
