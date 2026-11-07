/* Lightweight d3 chart components (thin marks, hairline grid, hover tooltips, table twins). */
(function (global) {
  "use strict";
  const tip = () => document.getElementById("tooltip");
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt1 = d3.format(",.1f"), fmt2 = d3.format(",.2f"), fmt3 = d3.format(".3f"), fmtPct = d3.format(".0%");

  function showTip(evt, title, rows) {
    const t = tip();
    t.innerHTML = `<div class="t-title">${title}</div>` + rows.map(([k, v]) => `<div class="t-row"><span class="t-key">${k}</span><span>${v}</span></div>`).join("");
    t.hidden = false;
    const pad = 14, r = t.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function hideTip() { tip().hidden = true; }

  function frame(el, margin) {
    const node = typeof el === "string" ? document.querySelector(el) : el;
    node.innerHTML = "";
    const W = node.clientWidth || 600, H = node.clientHeight || 260;
    const m = Object.assign({ top: 12, right: 16, bottom: 28, left: 44 }, margin || {});
    const svg = d3.select(node).append("svg").attr("width", W).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
    return { node, svg, g, w: W - m.left - m.right, h: H - m.top - m.bottom, m };
  }

  function empty(el, msg) { const n = typeof el === "string" ? document.querySelector(el) : el; n.innerHTML = `<div class="empty">${msg}</div>`; }

  /* ---------------- time series ---------------- */
  function timeSeries(el, opt) {
    const series = (opt.series || []).filter(s => s.points && s.points.some(p => Number.isFinite(p.value)));
    if (!series.length) return empty(el, opt.emptyText || "No data");
    const F = frame(el, opt.margin);
    const all = series.flatMap(s => s.points.filter(p => Number.isFinite(p.value)));
    const x = d3.scaleTime().domain(d3.extent(all, p => p.date)).range([0, F.w]);
    let ymax = d3.max(all, p => p.value);
    if (opt.band) ymax = Math.max(ymax, d3.max(opt.band.filter(b => Number.isFinite(b.hi)), b => b.hi) || 0);
    (opt.refLines || []).forEach(r => { if (r.y < ymax * 1.6) ymax = Math.max(ymax, r.y * 1.08); });
    const y = d3.scaleLinear().domain([0, ymax * 1.05]).nice().range([F.h, 0]);
    F.g.append("g").attr("class", "gridline").call(d3.axisLeft(y).ticks(5).tickSize(-F.w).tickFormat(""));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(F.w / 90))).tickSizeOuter(0));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    if (opt.yLabel) F.g.append("text").attr("class", "ref-label").attr("x", 0).attr("y", -2).text(opt.yLabel);
    if (opt.band) {
      const b = opt.band.filter(d => Number.isFinite(d.lo) && Number.isFinite(d.hi));
      F.g.append("path").datum(b).attr("fill", css("--band")).attr("d", d3.area().x(d => x(d.date)).y0(d => y(d.lo)).y1(d => y(d.hi)).defined(d => Number.isFinite(d.lo)));
    }
    (opt.refLines || []).forEach(r => {
      if (y(r.y) < 0) return;
      F.g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", F.w).attr("y1", y(r.y)).attr("y2", y(r.y));
      F.g.append("text").attr("class", "ref-label").attr("x", F.w - 2).attr("y", y(r.y) - 3).attr("text-anchor", "end").text(r.label);
    });
    series.forEach(s => {
      const line = d3.line().defined(p => Number.isFinite(p.value)).x(p => x(p.date)).y(p => y(p.value));
      if (s.dots) {
        F.g.append("g").selectAll("circle").data(s.points.filter(p => Number.isFinite(p.value))).join("circle")
          .attr("cx", p => x(p.date)).attr("cy", p => y(p.value)).attr("r", 2.2).attr("fill", s.color).attr("opacity", 0.75);
      } else {
        F.g.append("path").datum(s.points).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.width || 2)
          .attr("stroke-linejoin", "round").attr("d", line);
      }
    });
    // crosshair + tooltip
    const cross = F.g.append("line").attr("class", "crosshair").attr("y1", 0).attr("y2", F.h).attr("visibility", "hidden");
    const bis = d3.bisector(p => p.date).center;
    F.g.append("rect").attr("width", F.w).attr("height", F.h).attr("fill", "transparent")
      .on("mousemove", (evt) => {
        const [mx] = d3.pointer(evt);
        const d0 = x.invert(mx);
        const ref = series[0].points;
        const i = bis(ref, d0);
        if (i < 0 || !ref[i]) return;
        const dt = ref[i].date;
        cross.attr("x1", x(dt)).attr("x2", x(dt)).attr("visibility", "visible");
        const rows = series.map(s => {
          const j = bis(s.points, dt); const p = s.points[j];
          return [s.name, p && Math.abs(p.date - dt) < 864e5 * 1.5 && Number.isFinite(p.value) ? fmt1(p.value) : "—"];
        });
        if (opt.band) { const j = bis(opt.band, dt); const b = opt.band[j]; if (b && Number.isFinite(b.lo)) rows.push(["90% interval", `${fmt1(b.lo)} – ${fmt1(b.hi)}`]); }
        showTip(evt, d3.timeFormat("%d %b %Y")(dt), rows);
      })
      .on("mouseleave", () => { cross.attr("visibility", "hidden"); hideTip(); })
      .on("click", (evt) => { if (opt.onClick) { const [mx] = d3.pointer(evt); opt.onClick(x.invert(mx)); } });
  }

  /* ---------------- horizontal bars ---------------- */
  function hbars(el, rows, opt = {}) {
    if (!rows.length) return empty(el, "No data");
    const labelW = opt.labelWidth || 150;
    const F = frame(el, { left: labelW, right: 44, top: 6, bottom: 24 });
    const x = d3.scaleLinear().domain([opt.min ?? Math.min(0, d3.min(rows, r => r.value)), opt.max ?? d3.max(rows, r => r.value)]).nice().range([0, F.w]);
    const y = d3.scaleBand().domain(rows.map(r => r.label)).range([0, F.h]).padding(0.28);
    F.g.append("g").attr("class", "gridline").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).ticks(5).tickSize(-F.h).tickFormat(""));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0).tickFormat(opt.tickFormat || null));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(6)).select(".domain").remove();
    const bw = Math.min(18, y.bandwidth());
    F.g.append("g").selectAll("rect").data(rows).join("rect")
      .attr("x", r => x(Math.min(0, r.value))).attr("y", r => y(r.label) + (y.bandwidth() - bw) / 2)
      .attr("width", r => Math.max(1, Math.abs(x(r.value) - x(0)))).attr("height", bw).attr("rx", 3)
      .attr("fill", r => r.color || css("--series-1"))
      .on("mousemove", (evt, r) => showTip(evt, r.label, r.tip || [["value", (opt.valueFormat || fmt3)(r.value)]]))
      .on("mouseleave", hideTip);
    F.g.append("g").selectAll("text").data(rows.filter(r => r.direct)).join("text")
      .attr("x", r => x(Math.max(0, r.value)) + 4).attr("y", r => y(r.label) + y.bandwidth() / 2 + 4)
      .attr("class", "ref-label").text(r => (opt.valueFormat || fmt3)(r.value));
  }

  /* ---------------- density scatter (pre-binned in log space) ---------------- */
  function densityScatter(el, bins, opt = {}) {
    if (!bins || !bins.cells || !bins.cells.length) return empty(el, "No data");
    const F = frame(el, { left: 48, bottom: 36, right: 70 });
    const lo = bins.min, hi = bins.max;
    const x = d3.scaleLog().domain([lo, hi]).range([0, F.w]);
    const y = d3.scaleLog().domain([lo, hi]).range([F.h, 0]);
    const ticks = [2, 5, 10, 20, 50, 100, 200, 500].filter(t => t >= lo && t <= hi);
    F.g.append("g").attr("class", "gridline").call(d3.axisLeft(y).tickValues(ticks).tickSize(-F.w).tickFormat(""));
    F.g.append("g").attr("class", "gridline").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).tickValues(ticks).tickSize(-F.h).tickFormat(""));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).tickValues(ticks).tickFormat(d3.format("d")));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickValues(ticks).tickFormat(d3.format("d")));
    F.g.append("text").attr("class", "ref-label").attr("x", F.w / 2).attr("y", F.h + 32).attr("text-anchor", "middle").text("Observed PM2.5 (µg/m³)");
    F.g.append("text").attr("class", "ref-label").attr("transform", `translate(-36,${F.h / 2}) rotate(-90)`).attr("text-anchor", "middle").text("Predicted PM2.5 (µg/m³)");
    const maxc = d3.max(bins.cells, c => c[2]);
    const col = d3.scaleSequentialLog([1, maxc], t => d3.interpolateRgb(css("--accent-soft"), css("--accent"))(t));
    const n = bins.n, step = (Math.log10(hi) - Math.log10(lo)) / n;
    const cw = F.w / n, ch = F.h / n;
    F.g.append("g").selectAll("rect").data(bins.cells).join("rect")
      .attr("x", c => c[0] * cw).attr("y", c => F.h - (c[1] + 1) * ch).attr("width", cw + 0.3).attr("height", ch + 0.3)
      .attr("fill", c => col(c[2]))
      .on("mousemove", (evt, c) => showTip(evt, "Station-days", [["observed", `${fmt1(10 ** (Math.log10(lo) + c[0] * step))}–${fmt1(10 ** (Math.log10(lo) + (c[0] + 1) * step))}`], ["predicted", `${fmt1(10 ** (Math.log10(lo) + c[1] * step))}–${fmt1(10 ** (Math.log10(lo) + (c[1] + 1) * step))}`], ["count", c[2]]]))
      .on("mouseleave", hideTip);
    F.g.append("line").attr("class", "ref-line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(hi)).attr("y2", y(hi));
    F.g.append("text").attr("class", "ref-label").attr("x", x(hi) - 4).attr("y", y(hi) + 12).attr("text-anchor", "end").text("1:1");
    // colour legend
    const lg = F.g.append("g").attr("transform", `translate(${F.w + 14},0)`);
    const gh = Math.min(140, F.h);
    const gradId = "dens" + Math.random().toString(36).slice(2);
    const grad = F.svg.append("defs").append("linearGradient").attr("id", gradId).attr("x1", 0).attr("x2", 0).attr("y1", 1).attr("y2", 0);
    d3.range(0, 1.01, 0.1).forEach(t => grad.append("stop").attr("offset", t).attr("stop-color", col(Math.exp(Math.log(maxc) * t))));
    lg.append("rect").attr("width", 10).attr("height", gh).attr("fill", `url(#${gradId})`).attr("rx", 2);
    lg.append("text").attr("class", "ref-label").attr("x", 14).attr("y", 8).text(maxc);
    lg.append("text").attr("class", "ref-label").attr("x", 14).attr("y", gh).text("1");
    lg.append("text").attr("class", "ref-label").attr("x", 0).attr("y", gh + 14).text("count");
  }

  /* ---------------- correlogram ---------------- */
  function correlogram(el, cg) {
    if (!cg || !cg.bins_km) return empty(el, "No data");
    const F = frame(el, { left: 44, bottom: 34 });
    const x = d3.scaleLinear().domain([0, d3.max(cg.bins_km) * 1.05]).range([0, F.w]);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(cg.corr)), 1]).nice().range([F.h, 0]);
    F.g.append("g").attr("class", "gridline").call(d3.axisLeft(y).ticks(5).tickSize(-F.w).tickFormat(""));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.h})`).call(d3.axisBottom(x).ticks(6));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    F.g.append("text").attr("class", "ref-label").attr("x", F.w / 2).attr("y", F.h + 30).attr("text-anchor", "middle").text("Separation distance (km)");
    const xs = d3.range(0.5, x.domain()[1], 2);
    F.g.append("path").datum(xs).attr("fill", "none").attr("stroke", css("--series-2")).attr("stroke-width", 2)
      .attr("d", d3.line().x(d => x(d)).y(d => y(cg.c0 * Math.exp(-d / cg.L_km))));
    F.g.append("g").selectAll("circle").data(cg.bins_km.map((b, i) => ({ b, c: cg.corr[i], n: cg.npairs[i] }))).join("circle")
      .attr("cx", d => x(d.b)).attr("cy", d => y(d.c)).attr("r", 4.5).attr("fill", css("--series-1")).attr("stroke", css("--surface-1")).attr("stroke-width", 2)
      .on("mousemove", (evt, d) => showTip(evt, `${fmt1(d.b)} km`, [["correlation", fmt3(d.c)], ["pairs", d3.format(",")(d.n)]]))
      .on("mouseleave", hideTip);
    F.g.append("text").attr("class", "ref-label").attr("x", F.w).attr("y", 10).attr("text-anchor", "end")
      .text(`fit: ${fmt2(cg.c0)}·exp(−d / ${fmt1(cg.L_km)} km)`);
  }

  /* ---------------- tables ---------------- */
  function table(el, columns, rows, opt = {}) {
    const node = typeof el === "string" ? document.querySelector(el) : el;
    let sortKey = opt.sortKey, sortDir = opt.sortDir || -1;
    function render() {
      const data = rows.slice();
      if (sortKey) data.sort((a, b) => { const va = a[sortKey], vb = b[sortKey]; if (va == null) return 1; if (vb == null) return -1; return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir; });
      const head = columns.map(c => `<th class="${c.num ? "num " : ""}sortable" data-k="${c.key}">${c.label}${sortKey === c.key ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
      const body = data.map((r, i) => `<tr class="${opt.onRow ? "clickable" : ""} ${opt.rowClass ? opt.rowClass(r) : ""}" data-i="${rows.indexOf(r)}">` + columns.map(c => {
        const v = r[c.key];
        const txt = c.fmt ? c.fmt(v, r) : (v == null ? "—" : v);
        return `<td class="${c.num ? "num" : ""}">${txt}</td>`;
      }).join("") + "</tr>").join("");
      node.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      node.querySelectorAll("th.sortable").forEach(th => th.onclick = () => { const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : -1; sortKey = k; render(); });
      if (opt.onRow) node.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => { node.querySelectorAll("tr.selected").forEach(t => t.classList.remove("selected")); tr.classList.add("selected"); opt.onRow(rows[+tr.dataset.i]); });
    }
    render();
  }

  global.GHCharts = { timeSeries, hbars, densityScatter, correlogram, table, showTip, hideTip, css, fmt1, fmt2, fmt3, fmtPct };
})(window);
