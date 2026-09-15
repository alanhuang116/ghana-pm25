/* GH-PM25 downloads: per-day GeoTIFF / CSV built in the browser, and the release archive listing. */
(function () {
  "use strict";
  const S = window.GHScales;
  const DATA = "data";
  const NODATA = -9999;
  const $ = (id) => document.getElementById(id);
  let manifest = null;

  /* ---------- minimal GeoTIFF writer: uncompressed, chunky float32, EPSG:4326 ---------- */
  function geotiff(bands, w, h, lon0, lat1, res, names) {
    const nb = bands.length;
    const pixBytes = w * h * nb * 4;
    const tags = []; // [tag, type, count, valueArray]  types: 2 ASCII, 3 SHORT, 4 LONG, 12 DOUBLE
    const nodataStr = `${NODATA}\0`;
    const desc = `GH-PM25 daily PM2.5 (ug/m3): ${names.join(", ")}\0`;
    tags.push([256, 4, 1, [w]], [257, 4, 1, [h]], [258, 3, nb, Array(nb).fill(32)], [259, 3, 1, [1]], [262, 3, 1, [1]],
      [270, 2, desc.length, desc], [273, 4, 1, [0]], [277, 3, 1, [nb]], [278, 4, 1, [h]], [279, 4, 1, [pixBytes]],
      [284, 3, 1, [1]], [338, 3, nb - 1, Array(nb - 1).fill(0)], [339, 3, nb, Array(nb).fill(3)],
      [33550, 12, 3, [res, res, 0]], [33922, 12, 6, [0, 0, 0, lon0, lat1, 0]],
      [34735, 3, 16, [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]],
      [42113, 2, nodataStr.length, nodataStr]);
    if (nb === 1) tags.splice(tags.findIndex(t => t[0] === 338), 1);
    tags.sort((a, b) => a[0] - b[0]);
    const size = (t, c) => ({ 2: 1, 3: 2, 4: 4, 12: 8 }[t] * c);
    const ifdSize = 2 + tags.length * 12 + 4;
    let extra = 0;
    tags.forEach(([, t, c]) => { const s = size(t, c); if (s > 4) extra += s + (s % 2); });
    const ifdOff = 8, extraOff = ifdOff + ifdSize, pixOff = extraOff + extra;
    const buf = new ArrayBuffer(pixOff + pixBytes);
    const dv = new DataView(buf);
    dv.setUint16(0, 0x4949, true); dv.setUint16(2, 42, true); dv.setUint32(4, ifdOff, true);
    dv.setUint16(ifdOff, tags.length, true);
    let p = ifdOff + 2, xo = extraOff;
    for (const [tag, type, count, vals] of tags) {
      const v = tag === 273 ? [pixOff] : vals;
      dv.setUint16(p, tag, true); dv.setUint16(p + 2, type, true); dv.setUint32(p + 4, count, true);
      const s = size(type, count);
      let q = s > 4 ? xo : p + 8;
      if (s > 4) dv.setUint32(p + 8, xo, true);
      for (let i = 0; i < count; i++) {
        if (type === 2) { dv.setUint8(q, v.charCodeAt(i)); q += 1; }
        else if (type === 3) { dv.setUint16(q, v[i], true); q += 2; }
        else if (type === 4) { dv.setUint32(q, v[i], true); q += 4; }
        else if (type === 12) { dv.setFloat64(q, v[i], true); q += 8; }
      }
      if (s > 4) xo += s + (s % 2);
      p += 12;
    }
    dv.setUint32(p, 0, true);
    const px = new Float32Array(buf, pixOff, w * h * nb);
    for (let i = 0; i < w * h; i++) for (let b = 0; b < nb; b++) { const x = bands[b][i]; px[i * nb + b] = Number.isFinite(x) ? x : NODATA; }
    return new Blob([buf], { type: "image/tiff" });
  }

  function save(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
  }

  async function withDay(fn, btn) {
    const date = $("dl-date").value;
    if (!manifest.dates.includes(date)) { $("dl-status").textContent = `No map for ${date}. Choose a date between ${manifest.dates[0]} and ${manifest.dates[manifest.dates.length - 1]}.`; return; }
    btn.disabled = true; $("dl-status").textContent = `Preparing ${date}…`;
    try {
      const d = await S.loadDay(DATA, date);
      await fn(d, date);
      $("dl-status").textContent = `Downloaded ${date}.`;
    } catch (e) {
      $("dl-status").textContent = `Could not build the file for ${date}: ${e.message}`;
    } finally { btn.disabled = false; }
  }

  function toTif(d, date) {
    const g = manifest.grid;
    save(geotiff([d.pm, d.lo, d.hi], d.w, d.h, g.lon0, g.lat1, g.res, ["mean", "lower90", "upper90"]), `GH-PM25_1km_${date.replaceAll("-", "")}.tif`);
  }

  function toCsv(d, date) {
    const g = manifest.grid;
    const lines = ["lat,lon,pm25,lower90,upper90,cams_pm25,inside_aoa"];
    for (let r = 0; r < d.h; r++) {
      const lat = (g.lat1 - (r + 0.5) * g.res).toFixed(3);
      for (let c = 0; c < d.w; c++) {
        const i = r * d.w + c;
        if (d.code[i] === 0) continue;
        const lon = (g.lon0 + (c + 0.5) * g.res).toFixed(3);
        lines.push(`${lat},${lon},${d.pm[i].toFixed(1)},${d.lo[i].toFixed(1)},${d.hi[i].toFixed(1)},${d.cams[i]},${d.code[i] === 2 ? 1 : 0}`);
      }
    }
    save(new Blob([lines.join("\n")], { type: "text/csv" }), `GH-PM25_1km_${date.replaceAll("-", "")}.csv`);
  }

  const mb = (b) => `${(b / 1048576).toFixed(b > 10485760 ? 0 : 1)} MB`;

  function renderArchive(dl) {
    if (!dl || !dl.assets || !dl.assets.length) {
      $("dl-months").innerHTML = `<p class="muted">The archive is being published. Daily files for each day are available on the left.</p>`;
      $("dl-aggregates").innerHTML = "";
      return;
    }
    $("dl-archive-sub").innerHTML = `permanent files on <a href="${dl.release_url}" target="_blank" rel="noopener">GitHub Releases</a>`;
    const monthly = dl.assets.filter(a => a.kind === "daily_month").sort((a, b) => b.period.localeCompare(a.period));
    $("dl-months").innerHTML = `<table><thead><tr><th>Month</th><th class="num">Days</th><th class="num">Size</th><th></th></tr></thead><tbody>` +
      monthly.map(a => `<tr><td>${a.period}</td><td class="num">${a.days ?? "—"}</td><td class="num">${mb(a.size)}</td><td><a class="dl-link" href="${a.url}">Download zip</a></td></tr>`).join("") + "</tbody></table>";
    const agg = dl.assets.filter(a => a.kind !== "daily_month").sort((a, b) => (a.kind + b.period).localeCompare(b.kind + a.period));
    const lab = { annual_mean: "Annual mean GeoTIFF", monthly_mean: "Monthly mean GeoTIFF", region_stats: "Regional & district daily statistics (CSV)", readme: "Data dictionary (README)" };
    $("dl-aggregates").innerHTML = `<table><thead><tr><th>File</th><th>Period</th><th class="num">Size</th><th></th></tr></thead><tbody>` +
      agg.map(a => `<tr><td>${lab[a.kind] || a.name}</td><td>${a.period || "—"}</td><td class="num">${mb(a.size)}</td><td><a class="dl-link" href="${a.url}">Download</a></td></tr>`).join("") + "</tbody></table>";
    if (monthly[0]) $("dl-example-url").textContent = monthly[0].url;
  }

  async function init() {
    try { manifest = await fetch(`${DATA}/manifest.json`, { cache: "no-cache" }).then(r => r.json()); } catch (e) { return; }
    const di = $("dl-date");
    di.min = manifest.dates[0]; di.max = manifest.dates[manifest.dates.length - 1]; di.value = manifest.latest;
    const upd = () => { $("dl-tier").textContent = (manifest.tiers || {})[di.value] || (manifest.dates.includes(di.value) ? "final" : "no data"); };
    di.addEventListener("change", upd); upd();
    $("dl-tif").addEventListener("click", (e) => withDay(toTif, e.currentTarget));
    $("dl-csv").addEventListener("click", (e) => withDay(toCsv, e.currentTarget));
    const dl = await fetch(`${DATA}/downloads.json`, { cache: "no-cache" }).then(r => r.ok ? r.json() : null).catch(() => null);
    renderArchive(dl);
    // follow the monitor tab's date when the user switches tabs
    document.querySelector('.tabs button[data-tab="download"]').addEventListener("click", () => {
      const md = $("date-input").value; if (md && manifest.dates.includes(md)) { di.value = md; upd(); }
    });
  }
  init();
})();
