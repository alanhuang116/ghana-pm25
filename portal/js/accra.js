/* GH-PM25 Observatory - Accra city view (Greater Accra Region at full 1-km detail, plus TAHMO weather stations) */
(function () {
  "use strict";
  const S = window.GHScales, C = window.GHCharts, A = window.GHApp;
  const { state, $ } = A;
  const REGION = "Greater Accra Region";
  const VIEW = [[-0.52, 5.47], [0.72, 6.10]];   // Greater Accra extent
  const WX = { rh: ["Relative humidity", "%"], t2m: ["Air temperature", "°C"], precip: ["Rainfall", "mm/day"], ws10: ["Wind speed (≈2 m)", "m/s"] };
  const acc = { map: null, ready: false, sel: null, districts: [], adm1: -1, tahmo: null, built: false };
  const TS_IDS = { chart: "a-ts-chart", title: "a-ts-title", sub: "a-ts-sub", table: "a-ts-table" };

  A.on(async (what) => {
    if (what === "ready") acc.prepared = prepare();
    if (what === "tab:accra") { await acc.prepared; build(); }
    if (!acc.built) return;
    if (what === "draw") { updateImage(); A.renderLegend($("a-legend")); updateStations(); $("a-layer").value = state.layer; }
    if (what === "date") { syncDate(); renderKPIs(); renderDistricts(); if (acc.sel) A.renderSeriesPanel(true, TS_IDS, acc.sel); }
    if (what === "theme") { setBasemap(); renderWeather(); renderTrend(); }
  });

  /* ---------- data that does not need the map ---------- */
  async function prepare() {
    const R = state.regions;
    if (R) acc.adm1 = R.adm1.names.indexOf(REGION);
    const [adm1, adm2, tahmo] = await Promise.all([
      A.fetchJSON("boundaries/adm1.geojson").catch(() => null), A.fetchJSON("boundaries/adm2.geojson").catch(() => null),
      A.fetchJSON("accra/tahmo.json").catch(() => null)]);
    acc.tahmo = tahmo;
    const ga = adm1 && adm1.features.find(f => f.properties.name === REGION);
    acc.gaFeature = ga;
    if (ga && adm2 && R) {
      const inGA = adm2.features.filter(f => d3.geoContains(ga, d3.geoCentroid(f)));
      acc.adm2 = { type: "FeatureCollection", features: inGA };
      acc.districts = inGA.map(f => ({ name: f.properties.name, k: R.adm2.names.indexOf(f.properties.name) })).filter(d => d.k >= 0);
    }
  }

  /* ---------- build the tab on first open (the map needs a visible container) ---------- */
  function build() {
    if (acc.built) { setTimeout(() => acc.map && acc.map.resize(), 50); return; }
    if (!state.manifest) return;
    acc.built = true;
    initControls();
    initMap();
    syncDate(); renderKPIs(); renderDistricts(); initWeather(); renderTrend(); renderExperiments();
    A.renderSeriesPanel(false, TS_IDS, null);
    A.renderLegend($("a-legend"));
  }

  function initControls() {
    const m = state.manifest, di = $("a-date");
    di.min = m.dates[0]; di.max = m.dates[m.dates.length - 1];
    di.onchange = () => A.setDate(A.nearestDate(di.value));
    $("a-prev").onclick = () => A.stepDate(-1);
    $("a-next").onclick = () => A.stepDate(1);
    $("a-latest").onclick = () => A.setDate(m.latest);
    $("a-layer").value = ["pm25", "aqi", "who", "gepa", "relunc", "diff"].includes(state.layer) ? state.layer : "pm25";
    $("a-layer").onchange = (e) => { state.layer = e.target.value; $("layer-select").value = state.layer; A.drawLayer(); };
    $("a-opacity").oninput = (e) => acc.ready && acc.map.setPaintProperty("pm", "raster-opacity", +e.target.value);
    $("a-chk-stations").onchange = updateStations;
    $("a-chk-tahmo").onchange = () => acc.ready && acc.map.getLayer("tahmo") && acc.map.setLayoutProperty("tahmo", "visibility", $("a-chk-tahmo").checked ? "visible" : "none");
    window.addEventListener("resize", () => { if ($("tab-accra").classList.contains("active")) { renderWeather(); renderTrend(); } });
  }

  function syncDate() {
    $("a-date").value = state.date || "";
    const t = (state.manifest.tiers || {})[state.date] || "final";
    $("a-tier").textContent = t;
  }

  /* ---------- map ---------- */
  function initMap() {
    const g = state.manifest.grid;
    const map = new maplibregl.Map({ container: "map-accra", attributionControl: { compact: true }, style: { version: 8, sources: {}, layers: [] },
      bounds: VIEW, fitBoundsOptions: { padding: 10 }, maxZoom: 14, minZoom: 8 });
    acc.map = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("load", () => {
      map.addSource("base", { type: "raster", tiles: A.basemapTiles("nolabels"), tileSize: 256, maxzoom: 16, attribution: "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors" });
      map.addLayer({ id: "base", type: "raster", source: "base" });
      const blank = document.createElement("canvas"); blank.width = 2; blank.height = 2;
      map.addSource("pm", { type: "image", url: state.imageUrl || blank.toDataURL(), coordinates: [[g.lon0, g.lat1], [g.lon1, g.lat1], [g.lon1, g.lat0], [g.lon0, g.lat0]] });
      map.addLayer({ id: "pm", type: "raster", source: "pm", paint: { "raster-opacity": +$("a-opacity").value, "raster-resampling": "nearest", "raster-fade-duration": 0 } });
      map.addSource("labels", { type: "raster", tiles: A.basemapTiles("only_labels"), tileSize: 256 });
      map.addLayer({ id: "labels", type: "raster", source: "labels" });
      const line = A.isDark() ? "#d9d8d2" : "#3d3c39";
      if (acc.adm2) {
        map.addSource("districts", { type: "geojson", data: acc.adm2 });
        map.addLayer({ id: "districts-fill", type: "fill", source: "districts", paint: { "fill-opacity": 0 } });
        map.addLayer({ id: "districts-line", type: "line", source: "districts", paint: { "line-color": line, "line-width": 0.7, "line-opacity": 0.7 } });
      }
      if (acc.gaFeature) {
        map.addSource("region", { type: "geojson", data: acc.gaFeature });
        map.addLayer({ id: "region-line", type: "line", source: "region", paint: { "line-color": line, "line-width": 1.6 } });
      }
      map.addSource("stations", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: "stations", type: "circle", source: "stations", paint: {
        "circle-radius": ["case", ["get", "ref"], 8, 5.5], "circle-color": ["get", "color"], "circle-stroke-color": A.isDark() ? "#1a1a19" : "#ffffff",
        "circle-stroke-width": 2, "circle-opacity": ["case", ["get", "has"], 1, 0.35] } });
      if (acc.tahmo) {
        map.addSource("tahmo", { type: "geojson", data: { type: "FeatureCollection", features: acc.tahmo.stations.map(s => ({
          type: "Feature", geometry: { type: "Point", coordinates: [s.lon, s.lat] }, properties: { code: s.code, name: s.name } })) } });
        map.addLayer({ id: "tahmo", type: "circle", source: "tahmo", paint: { "circle-radius": 6, "circle-color": A.isDark() ? "#1a1a19" : "#ffffff",
          "circle-stroke-color": C.css("--series-3"), "circle-stroke-width": 3 } });
      }
      map.on("mousemove", onMove);
      map.on("mouseout", () => { $("a-readout").hidden = true; });
      map.on("click", onClick);
      for (const l of ["stations", "tahmo"]) {
        map.on("mouseenter", l, () => map.getCanvas().style.cursor = "pointer");
        map.on("mouseleave", l, () => map.getCanvas().style.cursor = "");
      }
      acc.ready = true;
      updateImage(); updateStations();
    });
  }
  function setBasemap() {
    if (!acc.ready) return;
    const m = acc.map, line = A.isDark() ? "#d9d8d2" : "#3d3c39";
    m.getSource("base").setTiles(A.basemapTiles("nolabels"));
    m.getSource("labels").setTiles(A.basemapTiles("only_labels"));
    for (const l of ["districts-line", "region-line"]) if (m.getLayer(l)) m.setPaintProperty(l, "line-color", line);
    m.setPaintProperty("stations", "circle-stroke-color", A.isDark() ? "#1a1a19" : "#ffffff");
    if (m.getLayer("tahmo")) m.setPaintProperty("tahmo", "circle-color", A.isDark() ? "#1a1a19" : "#ffffff");
  }
  function updateImage() {
    if (!acc.ready || !state.imageUrl) return;
    const g = state.manifest.grid;
    acc.map.getSource("pm").updateImage({ url: state.imageUrl, coordinates: [[g.lon0, g.lat1], [g.lon1, g.lat1], [g.lon1, g.lat0], [g.lon0, g.lat0]] });
  }
  function updateStations() {
    if (!acc.ready) return;
    acc.map.setLayoutProperty("stations", "visibility", $("a-chk-stations").checked ? "visible" : "none");
    acc.map.getSource("stations").setData({ type: "FeatureCollection", features: A.stationFeatures().filter(f => inView(f.geometry.coordinates)) });
  }
  const inView = ([lon, lat]) => lon >= VIEW[0][0] - 0.2 && lon <= VIEW[1][0] + 0.2 && lat >= VIEW[0][1] - 0.2 && lat <= VIEW[1][1] + 0.2;

  function onMove(e) {
    const d = state.day, ro = $("a-readout"); if (!d) return;
    const i = A.cellAt(e.lngLat.lng, e.lngLat.lat);
    if (i < 0 || d.code[i] === 0) { ro.hidden = true; return; }
    const pm = d.pm[i], cat = S.aqiCategory(pm), m = acc.map;
    const dist = m.getLayer("districts-fill") ? m.queryRenderedFeatures(e.point, { layers: ["districts-fill"] })[0] : null;
    const st = m.queryRenderedFeatures(e.point, { layers: ["stations"] })[0];
    const tw = m.getLayer("tahmo") ? m.queryRenderedFeatures(e.point, { layers: ["tahmo"] })[0] : null;
    let wx = "";
    if (tw) {
      const s = acc.tahmo.series[tw.properties.code], k = s ? s.date.indexOf(state.date) : -1;
      wx = `<div class="row"><span>${tw.properties.name} (TAHMO)</span><span>${k >= 0 && s.rh && s.rh[k] != null ? C.fmt1(s.rh[k]) + "% RH" : "no data"}</span></div>`;
    }
    ro.hidden = false;
    ro.innerHTML = `<div class="muted">${e.lngLat.lat.toFixed(3)}°N, ${e.lngLat.lng.toFixed(3)}°E${dist ? " · " + dist.properties.name : ""}</div>
      <div class="big">${C.fmt1(pm)} <span class="muted">µg/m³</span></div>
      <div class="row"><span>90% interval</span><span>${C.fmt1(d.lo[i])} – ${C.fmt1(d.hi[i])}</span></div>
      <div class="row"><span>AQI</span><span class="cat-chip"><i style="background:${cat.color}"></i>${S.aqiValue(pm)} · ${cat.label}</span></div>
      ${st ? `<div class="row"><span>${st.properties.name}</span><span>${st.properties.has ? C.fmt1(st.properties.obs) + " obs." : "no obs."}</span></div>` : ""}${wx}`;
  }
  function onClick(e) {
    const tw = acc.map.getLayer("tahmo") ? acc.map.queryRenderedFeatures(e.point, { layers: ["tahmo"] })[0] : null;
    if (tw) { $("a-wx-station").value = tw.properties.code; renderWeather(); }
    const st = acc.map.queryRenderedFeatures(e.point, { layers: ["stations"] })[0];
    acc.sel = { lng: e.lngLat.lng, lat: e.lngLat.lat, station: st ? st.properties.site_id : null };
    A.renderSeriesPanel(false, TS_IDS, acc.sel);
  }

  /* ---------- KPIs and district table ---------- */
  function tIndex() { return state.regions ? state.regions.dates.indexOf(state.date) : -1; }
  function renderKPIs() {
    const R = state.regions, t = tIndex(), k = acc.adm1;
    const obs = (state.obsByDate.get(state.date) || []);
    const sites = new Map(state.stations.map(s => [s.site_id, s]));
    const n = obs.filter(o => { const s = sites.get(o.site_id); return s && acc.gaFeature && d3.geoContains(acc.gaFeature, [s.lon, s.lat]); }).length;
    $("a-k-obs").textContent = n;
    if (!R || t < 0 || k < 0) { ["a-k-pw", "a-k-who", "a-k-gepa", "a-k-top"].forEach(id => $(id).textContent = "—"); return; }
    const a = R.adm1;
    $("a-k-pw").textContent = C.fmt1(a.pop_weighted[k][t]) + " µg/m³";
    $("a-k-pw-sub").textContent = `90% interval ${C.fmt1(a.lower90[k][t])}–${C.fmt1(a.upper90[k][t])} µg/m³`;
    $("a-k-who").textContent = C.fmtPct(a.pop_frac_gt15[k][t]);
    $("a-k-gepa").textContent = C.fmtPct(a.pop_frac_gt35[k][t]);
    let best = null;
    acc.districts.forEach(d => { const v = R.adm2.pop_weighted[d.k][t]; if (v != null && (!best || v > best.v)) best = { name: d.name, v }; });
    $("a-k-top").textContent = best ? best.name : "—";
    $("a-k-top-sub").textContent = best ? `${C.fmt1(best.v)} µg/m³ population-weighted` : "";
  }
  function renderDistricts() {
    const R = state.regions, t = tIndex();
    if (!R || t < 0 || !acc.districts.length) return C.table("#a-district-table", [], []);
    const a2 = R.adm2;
    const rows = acc.districts.map(d => ({ ...d, pw: a2.pop_weighted[d.k][t], lo: a2.lower90[d.k][t], hi: a2.upper90[d.k][t], f35: a2.pop_frac_gt35[d.k][t], pop: a2.population ? a2.population[d.k] : null }));
    C.table("#a-district-table", [
      { key: "name", label: "District" },
      { key: "pw", label: "PM2.5", num: true, fmt: (v) => v == null ? "—" : `<span class="cat-chip"><i style="background:${S.colorAt(S.PM_STOPS, v)}"></i>${C.fmt1(v)}</span>` },
      { key: "lo", label: "90% interval", num: true, fmt: (v, r) => r.lo == null ? "—" : `${C.fmt1(r.lo)}–${C.fmt1(r.hi)}` },
      { key: "f35", label: "Pop. > 35", num: true, fmt: (v) => v == null ? "—" : C.fmtPct(v) },
    ], rows, { sortKey: "pw", onRow: (r) => zoomTo(r.name) });
  }
  function zoomTo(name) {
    if (!acc.ready || !acc.adm2) return;
    const f = acc.adm2.features.find(x => x.properties.name === name); if (!f) return;
    const b = d3.geoBounds(f);
    acc.map.fitBounds(b, { padding: 40, maxZoom: 12 });
  }

  /* ---------- trend: Greater Accra vs Ghana ---------- */
  function renderTrend() {
    const R = state.regions; if (!R || acc.adm1 < 0 || !$("a-trend").clientWidth) return;
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
      $("a-wx-station").disabled = true; $("a-wx-var").disabled = true;
      $("a-wx-chart").innerHTML = `<div class="empty">No weather-station data published yet. It appears here after the TAHMO download runs with the project's credentials.</div>`;
      return;
    }
    // stations that stopped reporting are labelled; the default is the most recent one nearest the city centre
    const last = s => { const d = (T.series[s.code] || {}).date || []; return d.length ? d[d.length - 1] : ""; };
    const latest = d3.max(T.stations, last), old = s => (new Date(latest) - new Date(last(s) || 0)) / 864e5 > 30;
    $("a-wx-station").innerHTML = T.stations.map(s => `<option value="${s.code}">${s.name} (${s.code})${old(s) ? ` · until ${last(s) || "n/a"}` : ""}</option>`).join("");
    const dist = s => Math.hypot(s.lat - 5.60, (s.lon + 0.19) * Math.cos(5.6 * Math.PI / 180));  // to the centre used in scripts/12_tahmo_accra.py
    $("a-wx-station").value = T.stations.filter(s => !old(s)).sort((a, b) => dist(a) - dist(b)).concat(T.stations)[0].code;
    $("a-wx-station").onchange = renderWeather; $("a-wx-var").onchange = renderWeather;
    $("a-wx-sub").textContent = `${T.stations.length} TAHMO station${T.stations.length > 1 ? "s" : ""} · daily, updated ${T.generated_at.slice(0, 10)}`;
    renderWeather();
  }
  function renderWeather() {
    const T = acc.tahmo; if (!T || !T.stations.length || !$("a-wx-chart").clientWidth) return;
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
