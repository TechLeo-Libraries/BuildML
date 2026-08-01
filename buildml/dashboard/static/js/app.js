import { hydrateIcons, iconSvg } from "./icons.js";

const state = {
  meta: null,
  charts: null,
  chartsTheme: null,
  route: "cockpit",
  severity: "all",
  sectionFilter: "",
  sort: { key: null, dir: "asc" },
};

const main = () => document.getElementById("main");
const offlineBundle = () => window.__BUILDML_OFFLINE__ || null;

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

async function api(path) {
  const offline = offlineBundle();
  if (offline) {
    return offlineApi(path, offline);
  }
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${path}`);
  }
  return response.json();
}

function offlineApi(path, bundle) {
  const url = new URL(path, "http://offline.local");
  const pathname = url.pathname;
  if (pathname === "/api/meta") return structuredClone(bundle.meta);
  if (pathname === "/api/cockpit") return structuredClone(bundle.cockpit);
  if (pathname === "/api/charts") {
    const theme = url.searchParams.get("theme") || currentTheme();
    return structuredClone(theme === "dark" ? bundle.charts_dark : bundle.charts_light);
  }
  if (pathname.startsWith("/api/charts/")) {
    const chartId = decodeURIComponent(pathname.slice("/api/charts/".length));
    const theme = url.searchParams.get("theme") || currentTheme();
    const catalog = theme === "dark" ? bundle.charts_dark : bundle.charts_light;
    if (!catalog?.[chartId]) throw new Error(`Unknown chart: ${chartId}`);
    return { id: chartId, figure: structuredClone(catalog[chartId]) };
  }
  if (pathname.startsWith("/api/domains/")) {
    const key = decodeURIComponent(pathname.slice("/api/domains/".length));
    const payload = bundle.domains?.[key];
    if (!payload) throw new Error(`Unknown domain: ${key}`);
    return structuredClone(payload);
  }
  if (pathname === "/api/concepts") {
    const q = (url.searchParams.get("q") || "").trim().toLowerCase();
    let items = bundle.concepts || [];
    if (q) {
      items = items.filter(
        (item) =>
          item.key.includes(q) ||
          item.title.toLowerCase().includes(q) ||
          item.summary.toLowerCase().includes(q) ||
          (item.details || []).some((detail) => String(detail).toLowerCase().includes(q)),
      );
    }
    return { count: items.length, concepts: structuredClone(items) };
  }
  if (pathname.startsWith("/api/concepts/")) {
    const key = decodeURIComponent(pathname.slice("/api/concepts/".length));
    const detail = bundle.concept_details?.[key];
    if (!detail) throw new Error(`Unknown concept: ${key}`);
    return structuredClone(detail);
  }
  if (pathname === "/api/search") {
    const needle = (url.searchParams.get("q") || "").trim().toLowerCase();
    const findings = (bundle.report?.findings || []).filter((item) => {
      const blob = `${item.title || ""} ${item.detail || ""} ${item.key || ""}`.toLowerCase();
      return blob.includes(needle);
    });
    const concepts = (bundle.concepts || []).filter(
      (item) =>
        item.key.includes(needle) ||
        item.title.toLowerCase().includes(needle) ||
        item.summary.toLowerCase().includes(needle),
    );
    const domains = (bundle.meta?.domains || []).filter(
      (item) =>
        item.key.includes(needle) ||
        item.title.toLowerCase().includes(needle) ||
        item.short.toLowerCase().includes(needle),
    );
    return {
      query: url.searchParams.get("q") || "",
      findings: findings.slice(0, 30),
      concepts: concepts.slice(0, 30),
      domains,
    };
  }
  if (pathname.startsWith("/api/export/")) {
    throw new Error("CSV/PDF export requires the live Teaching Studio server.");
  }
  throw new Error(`Offline snapshot has no route for ${path}`);
}

function toast(message) {
  const el = document.getElementById("toast");
  el.hidden = false;
  el.textContent = message;
  window.clearTimeout(toast._t);
  toast._t = window.setTimeout(() => {
    el.hidden = true;
  }, 2600);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("buildml-eda-theme", theme);
  const btn = document.getElementById("theme-toggle");
  btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  btn.querySelector("[data-icon]")?.setAttribute("data-icon", theme === "dark" ? "sun" : "moon");
  hydrateIcons(btn);
  // Rebuild figures from the theme-specific catalog so ink, series, heatmaps,
  // gauges, and annotations all follow the SPA theme.
  state.charts = null;
  state.chartsTheme = null;
  if (document.querySelector("[data-chart-id]")) {
    void restyleCharts();
  }
}

function themeTokens() {
  const dark = currentTheme() === "dark";
  return {
    dark,
    ink: dark ? "#E8EEF5" : "#1C2430",
    muted: dark ? "#9AA8B8" : "#5B6775",
    grid: dark ? "#2A3340" : "#D7DEE7",
    hoverBg: dark ? "#171C24" : "#FFFFFF",
    accent: dark ? "#3DDC97" : "#0B6E4F",
    critical: dark ? "#F07178" : "#B91C1C",
    series: dark
      ? ["#3DDC97", "#7DB4F0", "#F0A35A", "#C4B5FD", "#5EEAD4", "#FB7185", "#94A3B8", "#FBBF24"]
      : ["#0B6E4F", "#1D4E89", "#B45309", "#6D28D9", "#0F766E", "#9F1239", "#334155", "#A16207"],
    heatmap: dark
      ? [
          [0.0, "#7DB4F0"],
          [0.5, "#1D2430"],
          [1.0, "#3DDC97"],
        ]
      : [
          [0.0, "#1D4E89"],
          [0.5, "#F8FAFC"],
          [1.0, "#0B6E4F"],
        ],
  };
}

function chartLayoutOverrides() {
  const t = themeTokens();
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: "Segoe UI, Helvetica Neue, sans-serif",
      color: t.ink,
    },
    title: { font: { color: t.ink } },
    colorway: t.series,
    hoverlabel: {
      bgcolor: t.hoverBg,
      font: { size: 12, color: t.ink },
      bordercolor: t.grid,
    },
    legend: { font: { color: t.ink } },
    coloraxis: {
      colorbar: {
        tickfont: { color: t.ink },
        title: { font: { color: t.ink } },
      },
    },
    xaxis: {
      gridcolor: t.grid,
      zeroline: false,
      tickfont: { color: t.ink },
      title: { font: { color: t.ink } },
      linecolor: t.grid,
    },
    yaxis: {
      gridcolor: t.grid,
      zeroline: false,
      tickfont: { color: t.ink },
      title: { font: { color: t.ink } },
      linecolor: t.grid,
    },
    meta: {
      buildml_theme: t.dark ? "dark" : "light",
      annotation_color: t.muted,
      heatmap: t.heatmap,
      accent: t.accent,
      critical: t.critical,
    },
  };
}

async function ensureCharts() {
  const theme = currentTheme();
  if (!state.charts || state.chartsTheme !== theme) {
    state.charts = await api(`/api/charts?theme=${encodeURIComponent(theme)}`);
    state.chartsTheme = theme;
  }
  return state.charts;
}

async function restyleCharts() {
  const charts = await ensureCharts();
  document.querySelectorAll("[data-chart-id]").forEach((host) => {
    const id = host.getAttribute("data-chart-id");
    const fig = charts?.[id];
    if (!fig || !window.Plotly) return;
    const layout = themedLayout(fig.layout);
    window.Plotly.react(host, themeTraceData(fig.data), layout, {
      displayModeBar: false,
      responsive: true,
    });
  });
}

function themedLayout(baseLayout) {
  const overrides = chartLayoutOverrides();
  const base = baseLayout || {};
  const layout = {
    ...base,
    ...overrides,
    title:
      typeof base.title === "object"
        ? { ...base.title, font: { ...(base.title.font || {}), color: overrides.font.color } }
        : { text: base.title, font: { color: overrides.font.color } },
    xaxis: { ...(base.xaxis || {}), ...overrides.xaxis },
    yaxis: { ...(base.yaxis || {}), ...overrides.yaxis },
    legend: { ...(base.legend || {}), ...overrides.legend },
    hoverlabel: { ...(base.hoverlabel || {}), ...overrides.hoverlabel },
  };
  if (Array.isArray(base.annotations)) {
    const muted = overrides.meta.annotation_color;
    layout.annotations = base.annotations.map((item) => ({
      ...item,
      font: { ...(item.font || {}), color: muted },
    }));
  }
  // Theme multi-axis figures (Plotly may emit xaxis2/yaxis2…).
  for (const key of Object.keys(base)) {
    if (/^xaxis\d+$/.test(key)) {
      layout[key] = { ...base[key], ...overrides.xaxis };
    }
    if (/^yaxis\d+$/.test(key)) {
      layout[key] = { ...base[key], ...overrides.yaxis };
    }
  }
  return layout;
}

function themeTraceData(data) {
  const t = themeTokens();
  if (!Array.isArray(data)) return data;
  return data.map((trace, index) => {
    const next = { ...trace };
    if (next.type === "heatmap" || next.type === "heatmapgl") {
      next.colorscale = t.heatmap;
      next.colorbar = {
        ...(next.colorbar || {}),
        tickfont: { color: t.ink },
        title: {
          ...((next.colorbar && next.colorbar.title) || {}),
          font: { color: t.ink },
        },
      };
    }
    if (next.type === "indicator" && next.gauge) {
      next.gauge = {
        ...next.gauge,
        bar: { ...(next.gauge.bar || {}), color: t.accent },
        threshold: {
          ...(next.gauge.threshold || {}),
          line: { ...((next.gauge.threshold && next.gauge.threshold.line) || {}), color: t.critical },
        },
      };
      next.number = { ...(next.number || {}), font: { ...((next.number && next.number.font) || {}), color: t.ink } };
      next.title = {
        ...(next.title || {}),
        font: { ...((next.title && next.title.font) || {}), color: t.ink },
      };
    }
    if (next.marker && typeof next.marker === "object" && !Array.isArray(next.marker.color)) {
      // Keep explicit multi-color arrays from the server; retint single accent markers.
      if (typeof next.marker.color === "string" && (next.marker.color.startsWith("#") || next.marker.color.startsWith("rgb"))) {
        // leave categorical/series colors from themed catalog
      }
    }
    if (next.line && typeof next.line.color === "string") {
      // catalog already theme-aware
    }
    if (!next.marker && (next.type === "bar" || next.type === "scatter") && !next.marker_color) {
      next.marker = { color: t.series[index % t.series.length] };
    }
    return next;
  });
}

async function renderChart(host, chartId) {
  host.setAttribute("data-chart-id", chartId);
  host.innerHTML = `<div class="empty">Loading chart…</div>`;
  try {
    const charts = await ensureCharts();
    const fig = charts[chartId];
    if (!fig) {
      host.innerHTML = `<div class="empty">Chart unavailable.</div>`;
      return;
    }
    if (!window.Plotly) {
      host.innerHTML = `<div class="error-state">Plotly failed to load. Reinstall buildml[dashboard].</div>`;
      return;
    }
    host.innerHTML = "";
    await window.Plotly.newPlot(
      host,
      themeTraceData(fig.data),
      themedLayout(fig.layout),
      { displayModeBar: false, responsive: true },
    );
    window.Plotly.Plots.resize(host);
  } catch (error) {
    host.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function badge(severity) {
  const s = String(severity || "info").toLowerCase();
  return `<span class="badge ${escapeHtml(s)}">${escapeHtml(s)}</span>`;
}

function csvButtons(sections) {
  if (!sections?.length) return "";
  if (offlineBundle()) {
    return `<div class="chip-row"><span class="chip" title="CSV/PDF exports require the live Teaching Studio server">Offline · exports in live Studio</span></div>`;
  }
  return `<div class="chip-row">${sections
    .map(
      (key) =>
        `<a class="chip" href="/api/export/csv/${encodeURIComponent(key)}">${iconSvg("download")} ${escapeHtml(key.replaceAll("_", " "))}</a>`,
    )
    .join("")}</div>`;
}

function paragraphsHtml(text) {
  return String(text || "")
    .split(/\n\n+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => `<p>${escapeHtml(part)}</p>`)
    .join("");
}

function listHtml(items) {
  if (!items?.length) return "";
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function teachingHtml(studio) {
  if (!studio) return "";
  const concepts = (studio.concepts || [])
    .map((key) => `<button type="button" class="chip" data-concept="${escapeHtml(key)}">${escapeHtml(key)}</button>`)
    .join("");
  const worked = studio.worked_example || {};
  const next = studio.next_action || {};
  const thresholds = listHtml(studio.thresholds || studio.interpretation_thresholds);
  const assumptions = listHtml(studio.assumptions || []);
  const pitfalls = listHtml(studio.pitfalls || []);
  const interpretation = listHtml(studio.interpretation || []);
  const checklist = listHtml(studio.practice_checklist || []);
  const mastery = listHtml(studio.mastery_notes || []);
  return `
    <section class="panel teaching" aria-labelledby="studio-title">
      <div class="panel-head">
        <div>
          <h2 id="studio-title">Teaching Studio · ${escapeHtml(studio.title)}</h2>
          <p>What is analyzed, why, how it is computed, thresholds, pitfalls, and a worked example from this dataset.</p>
        </div>
        <span class="badge info">${iconSvg("book")} Studio</span>
      </div>
      <div class="teaching-scroll">
        <div class="teaching-block"><h3>What is analyzed</h3>${paragraphsHtml(studio.definition)}</div>
        <div class="teaching-block"><h3>Why it matters</h3>${paragraphsHtml(studio.why)}</div>
        <div class="teaching-block"><h3>How BuildML computes it</h3>${paragraphsHtml(studio.how)}</div>
        <div class="teaching-block"><h3>Interpretation rules</h3>${interpretation}</div>
        ${thresholds ? `<div class="teaching-block"><h3>Thresholds and review cues</h3>${thresholds}</div>` : ""}
        ${assumptions ? `<div class="teaching-block"><h3>Assumptions</h3>${assumptions}</div>` : ""}
        <div class="teaching-block"><h3>Pitfalls and anti-patterns</h3>${pitfalls}</div>
        <div class="worked">
          <h3>Worked example (this dataset)</h3>
          <p>${escapeHtml(worked.summary || "")}</p>
          <p>${escapeHtml(worked.reading || "")}</p>
          <pre class="mono" style="white-space:pre-wrap;margin:0.6rem 0 0;max-width:100%;overflow:auto">${escapeHtml(JSON.stringify(worked.values || {}, null, 2))}</pre>
        </div>
        <div class="teaching-block"><h3>Impact on modeling</h3>${paragraphsHtml(studio.modeling_impact)}</div>
        ${checklist ? `<div class="teaching-block checklist"><h3>Practice checklist</h3>${checklist}</div>` : ""}
        ${mastery ? `<div class="teaching-block"><h3>Mastery notes</h3>${mastery}</div>` : ""}
        <div class="teaching-block">
          <h3>Next action</h3>
          <p>${escapeHtml(next.label || "")}</p>
          <div class="api-chip mono">${escapeHtml(next.api || "")}</div>
        </div>
        <div class="teaching-block">
          <h3>Related concepts</h3>
          <div class="chip-row">${concepts}</div>
        </div>
      </div>
    </section>
  `;
}

function findingsHtml(findings) {
  const filtered = (findings || []).filter((item) => {
    if (state.severity !== "all" && String(item.severity).toLowerCase() !== state.severity) return false;
    if (!state.sectionFilter) return true;
    const blob = `${item.title} ${item.detail} ${item.key}`.toLowerCase();
    return blob.includes(state.sectionFilter.toLowerCase());
  });
  if (!filtered.length) {
    return `<div class="empty">No findings match the current filters.</div>`;
  }
  return `<div class="list">${filtered
    .map((item) => {
      const cols = (item.affected_columns || [])
        .slice(0, 8)
        .map((c) => `<span class="chip">${escapeHtml(c)}</span>`)
        .join("");
      return `
        <article class="finding" data-finding-key="${escapeHtml(item.key)}">
          <div class="chip-row">${badge(item.severity)}<span class="mono">${escapeHtml(item.key)}</span></div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.detail)}</p>
          <div class="finding-links">${cols}</div>
        </article>`;
    })
    .join("")}</div>`;
}

function tableFromRows(rows, tableId) {
  if (!rows?.length) return `<div class="empty">No tabular evidence in this section.</div>`;
  const keys = Object.keys(rows[0]);
  let sorted = [...rows];
  if (state.sort.key && keys.includes(state.sort.key)) {
    sorted.sort((a, b) => {
      const av = a[state.sort.key];
      const bv = b[state.sort.key];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return state.sort.dir === "asc" ? av - bv : bv - av;
      }
      return state.sort.dir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
  }
  if (state.sectionFilter) {
    const needle = state.sectionFilter.toLowerCase();
    sorted = sorted.filter((row) => JSON.stringify(row).toLowerCase().includes(needle));
  }
  return `
    <div class="table-wrap">
      <table class="data" id="${tableId}">
        <thead><tr>${keys
          .map(
            (key) =>
              `<th scope="col" data-sort-key="${escapeHtml(key)}" aria-sort="${
                state.sort.key === key ? (state.sort.dir === "asc" ? "ascending" : "descending") : "none"
              }">${escapeHtml(key)}</th>`,
          )
          .join("")}</tr></thead>
        <tbody>${sorted
          .slice(0, 200)
          .map(
            (row) =>
              `<tr>${keys
                .map((key) => `<td>${escapeHtml(formatCell(row[key]))}</td>`)
                .join("")}</tr>`,
          )
          .join("")}</tbody>
      </table>
    </div>
    <p class="page-sub">Showing ${Math.min(sorted.length, 200)} of ${sorted.length} rows.</p>
  `;
}

function formatCell(value) {
  if (value == null) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(4);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function chartCards(ids) {
  return `<div class="chart-grid">${ids
    .map(
      (id) => `
      <section class="panel chart-card">
        <div class="chart-actions">
          <button type="button" class="icon-btn" data-expand-chart="${escapeHtml(id)}" aria-label="Expand figure">
            ${iconSvg("expand")}
          </button>
        </div>
        <div class="chart-host" data-mount-chart="${escapeHtml(id)}"></div>
      </section>`,
    )
    .join("")}</div>`;
}

async function mountCharts(root) {
  const hosts = [...root.querySelectorAll("[data-mount-chart]")];
  await Promise.all(hosts.map((host) => renderChart(host, host.getAttribute("data-mount-chart"))));
}

function rowsFromSection(section) {
  if (!section) return [];
  if (Array.isArray(section)) return section.filter((item) => item && typeof item === "object");
  if (typeof section !== "object") return [];

  // Analyzer truth: univariate/outliers expose per_column maps.
  if (section.per_column && typeof section.per_column === "object" && !Array.isArray(section.per_column)) {
    return Object.entries(section.per_column).map(([column, stats]) => {
      if (stats && typeof stats === "object" && !Array.isArray(stats)) {
        const flat = { column };
        for (const [key, value] of Object.entries(stats)) {
          if (key === "iqr_bounds" && Array.isArray(value) && value.length === 2) {
            flat.iqr_lower = value[0];
            flat.iqr_upper = value[1];
          } else if (key === "top_values" && value && typeof value === "object") {
            const topKey = Object.keys(value)[0];
            flat.top_value = topKey;
            flat.top_count = value[topKey];
          } else if (value == null || typeof value !== "object") {
            flat[key] = value;
          }
        }
        return flat;
      }
      return { column, value: stats };
    });
  }

  // Prefer nested tabular maps
  for (const key of [
    "vif",
    "numeric",
    "categorical",
    "flagged_columns",
    "missing_rate_by_column",
    "mutual_information_vs_target",
  ]) {
    if (key in section) {
      const value = section[key];
      if (Array.isArray(value)) return value.filter((item) => typeof item === "object");
      if (value && typeof value === "object") {
        return Object.entries(value).map(([k, v]) =>
          typeof v === "object" && v && !Array.isArray(v) ? { key: k, ...flattenShallow(v) } : { key: k, value: v },
        );
      }
    }
  }
  return Object.entries(section)
    .filter(([, v]) => typeof v !== "object" || v === null)
    .map(([key, value]) => ({ key, value }));
}

function flattenShallow(value) {
  const out = {};
  for (const [key, nested] of Object.entries(value)) {
    if (nested == null || typeof nested !== "object") out[key] = nested;
  }
  return out;
}

async function renderCockpit() {
  const data = await api("/api/cockpit");
  const overview = data.overview || {};
  const readiness = data.readiness || {};
  document.getElementById("filter-bar").hidden = false;
  main().innerHTML = `
    <section class="hero-strip">
      <article class="panel">
        <div class="panel-head">
          <div>
            <h2>Readiness cockpit</h2>
            <p>Sampling, roles, and severity before you change the pipeline.</p>
          </div>
          <span class="badge ${escapeHtml(readiness.status || "review")}">${escapeHtml(readiness.status || "review")}</span>
        </div>
        <div class="metric-grid">
          <div class="metric"><div class="label">Rows</div><div class="value">${escapeHtml(overview.n_rows)}</div></div>
          <div class="metric"><div class="label">Analysis rows</div><div class="value">${escapeHtml(overview.analysis_rows)}</div></div>
          <div class="metric"><div class="label">Completeness</div><div class="value">${formatPct(data.quality?.completeness_score)}</div></div>
          <div class="metric"><div class="label">Blocking findings</div><div class="value">${escapeHtml(readiness.blocking_findings)}</div></div>
          <div class="metric"><div class="label">Engine</div><div class="value">${escapeHtml(overview.engine || "pandas")}</div></div>
          <div class="metric"><div class="label">Lazy native</div><div class="value">${escapeHtml(overview.has_lazy_native ? "yes" : "no")}</div></div>
        </div>
        <div class="list" style="margin-top:1rem">
          ${(overview.engine_disclosures || []).map((w) => `<div class="finding">${iconSvg("alert")} ${escapeHtml(w)}</div>`).join("")}
          ${(data.warnings || []).map((w) => `<div class="finding">${iconSvg("alert")} ${escapeHtml(w)}</div>`).join("") || '<div class="empty">No sampling warnings.</div>'}
        </div>
      </article>
      <article class="panel">
        <div class="panel-head"><div><h2>Next actions</h2><p>Recommendations cite finding keys.</p></div></div>
        <div class="list">
          ${(readiness.next_actions || [])
            .map(
              (item) => `
              <div class="rec">
                <div class="chip-row">${badge(item.priority || "next")}</div>
                <h3>${escapeHtml(item.title)}</h3>
                <p>${escapeHtml(item.rationale)}</p>
                ${item.api ? `<div class="api-chip mono">session.${escapeHtml(item.api)}(...)</div>` : ""}
              </div>`,
            )
            .join("") || '<div class="empty">No recommendations produced.</div>'}
        </div>
      </article>
    </section>
    ${chartCards(data.chart_ids || [])}
    <section class="split-2">
      <article class="panel">
        <div class="panel-head">
          <div><h2>Key findings</h2><p>Drill from claim to affected columns.</p></div>
          ${csvButtons(["findings", "recommendations", "roles"])}
        </div>
        ${findingsHtml(data.findings)}
      </article>
      ${teachingHtml(data.teaching)}
    </section>
  `;
  await mountCharts(main());
  wireConceptChips(main());
}

async function renderDomain(domainKey) {
  const data = await api(`/api/domains/${domainKey}`);
  document.getElementById("filter-bar").hidden = false;
  const sections = data.sections || {};
  const tables = Object.entries(sections)
    .map(([name, payload]) => {
      const rows = rowsFromSection(payload);
      return `
        <article class="panel">
          <div class="panel-head">
            <div><h2>${escapeHtml(name)}</h2><p>Numeric detail kept visible.</p></div>
          </div>
          ${tableFromRows(rows, `table-${name}`)}
        </article>`;
    })
    .join("");
  main().innerHTML = `
    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>${escapeHtml(data.domain.title)}</h2>
          <p>${escapeHtml(data.domain.short)}</p>
        </div>
        ${csvButtons(data.domain.csv_sections || [])}
      </div>
    </section>
    ${chartCards(data.chart_ids || [])}
    <section class="split-2">
      <div class="list">${tables || '<div class="empty">No domain tables.</div>'}</div>
      ${teachingHtml(data.teaching)}
    </section>
    <section class="panel">
      <div class="panel-head"><div><h2>Domain findings</h2></div></div>
      ${findingsHtml(data.findings)}
    </section>
  `;
  await mountCharts(main());
  wireConceptChips(main());
  wireTableSort(main());
}

async function renderAcademy(query = "") {
  document.getElementById("filter-bar").hidden = true;
  const data = await api(`/api/concepts${query ? `?q=${encodeURIComponent(query)}` : ""}`);
  main().innerHTML = `
    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Concept Academy</h2>
          <p>Searchable notes for roles, leakage, partitions, MI, VIF, PCA, drift, and more.</p>
        </div>
        ${csvButtons(["concepts"])}
      </div>
      <label class="search-field" style="max-width:520px">
        ${iconSvg("search")}
        <input id="academy-search" type="search" placeholder="Search concepts" value="${escapeHtml(query)}" />
      </label>
      <p class="page-sub" style="margin-top:0.75rem">${data.count} concepts</p>
    </section>
    <section class="chart-grid">
      ${(data.concepts || [])
        .map(
          (item) => `
          <article class="concept-card panel">
            <div class="chip-row"><span class="badge info">${iconSvg("graduation-cap")} concept</span></div>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.summary)}</p>
            <div class="chip-row" style="margin-top:0.8rem">
              <button type="button" class="chip" data-concept="${escapeHtml(item.key)}">Open</button>
              ${(item.related_concepts || [])
                .slice(0, 4)
                .map((key) => `<button type="button" class="chip" data-concept="${escapeHtml(key)}">${escapeHtml(key)}</button>`)
                .join("")}
            </div>
          </article>`,
        )
        .join("")}
    </section>
  `;
  const input = document.getElementById("academy-search");
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      renderAcademy(input.value.trim());
    }
  });
  wireConceptChips(main());
}

function conceptSection(title, itemsOrText) {
  if (!itemsOrText) return "";
  if (Array.isArray(itemsOrText)) {
    if (!itemsOrText.length) return "";
    return `<div class="drawer-section"><h3>${escapeHtml(title)}</h3>${listHtml(itemsOrText)}</div>`;
  }
  const text = String(itemsOrText).trim();
  if (!text) return "";
  return `<div class="drawer-section"><h3>${escapeHtml(title)}</h3>${paragraphsHtml(text)}</div>`;
}

async function openConcept(key) {
  const data = await api(`/api/concepts/${encodeURIComponent(key)}`);
  const note = data.concept || {};
  const drawer = document.getElementById("concept-drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  document.getElementById("drawer-title").textContent = note.title || key;
  document.getElementById("drawer-body").innerHTML = `
    <div class="drawer-section">
      <p><strong>${escapeHtml(note.summary || "")}</strong></p>
    </div>
    ${conceptSection("Definition", note.definition || note.details?.slice?.(0, 1)?.[0])}
    ${conceptSection("Intuition", note.intuition)}
    ${conceptSection("Formal idea", note.formal_idea)}
    ${conceptSection("Why it matters in ML workflows", note.why_it_matters)}
    ${conceptSection("How BuildML uses it", note.how_buildml_uses)}
    ${conceptSection("How to interpret outputs", note.interpretation_rules || note.interpretation)}
    ${conceptSection("Assumptions", note.assumptions)}
    ${conceptSection("Failure modes", note.failure_modes)}
    ${conceptSection("Anti-patterns", note.anti_patterns)}
    ${conceptSection("Worked example pattern", note.worked_example_pattern)}
    ${
      !note.intuition && !note.formal_idea && (note.details || []).length
        ? `<div class="drawer-section"><h3>Details</h3>${listHtml(note.details)}</div>`
        : ""
    }
    <div class="drawer-section">
      <h3>Related concepts</h3>
      <div class="chip-row">${(data.related || [])
        .map((item) => `<button type="button" class="chip" data-concept="${escapeHtml(item.key)}">${escapeHtml(item.title)}</button>`)
        .join("") || '<span class="page-sub">None listed.</span>'}</div>
    </div>
    <div class="drawer-section">
      <h3>Linked studios</h3>
      <div class="chip-row">${(data.linked_domains || [])
        .map((domain) => `<a class="chip" href="#/${escapeHtml(domain)}">${escapeHtml(domain)}</a>`)
        .join("") || '<span class="page-sub">No studio links.</span>'}</div>
    </div>
  `;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  wireConceptChips(document.getElementById("drawer-body"));
  drawer.querySelector(".drawer-body")?.scrollTo?.(0, 0);
}

function closeConcept() {
  const drawer = document.getElementById("concept-drawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  document.getElementById("drawer-backdrop").hidden = true;
  document.body.style.overflow = "";
}

function setMobileNavOpen(open) {
  const sidebar = document.querySelector(".sidebar");
  const scrim = document.getElementById("mobile-nav-scrim");
  sidebar?.classList.toggle("open", open);
  if (scrim) {
    scrim.hidden = !open;
    scrim.classList.toggle("open", open);
  }
}

function wireConceptChips(root) {
  root.querySelectorAll("[data-concept]").forEach((node) => {
    node.addEventListener("click", () => openConcept(node.getAttribute("data-concept")));
  });
  root.querySelectorAll("[data-expand-chart]").forEach((node) => {
    node.addEventListener("click", async () => {
      const id = node.getAttribute("data-expand-chart");
      const charts = await ensureCharts();
      const fig = charts[id];
      const modal = document.getElementById("figure-modal");
      document.getElementById("modal-title").textContent = fig?.layout?.title?.text || id;
      const host = document.getElementById("modal-figure");
      host.innerHTML = "";
      modal.showModal();
      if (fig && window.Plotly) {
        await window.Plotly.newPlot(
          host,
          themeTraceData(fig.data),
          { ...themedLayout(fig.layout), height: 560 },
          { displayModeBar: true, responsive: true },
        );
      }
    });
  });
}

function wireTableSort(root) {
  root.querySelectorAll("th[data-sort-key]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort-key");
      if (state.sort.key === key) {
        state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
      } else {
        state.sort.key = key;
        state.sort.dir = "asc";
      }
      navigate(state.route);
    });
  });
}

function formatPct(value) {
  const n = Number(value);
  if (Number.isNaN(n)) return "n/a";
  return `${(n * 100).toFixed(1)}%`;
}

function renderNav() {
  const nav = document.getElementById("domain-nav");
  nav.innerHTML = (state.meta?.domains || [])
    .map((domain) => {
      const current = domain.key === state.route ? ' aria-current="page"' : "";
      return `<a class="nav-link" href="#/${escapeHtml(domain.key)}"${current}>
        <span>${iconSvg(domain.icon)}</span>
        <span class="nav-title">${escapeHtml(domain.title)}</span>
        <span class="nav-short">${escapeHtml(domain.short)}</span>
      </a>`;
    })
    .join("");
}

async function navigate(route) {
  state.route = route || "cockpit";
  const domain = (state.meta?.domains || []).find((item) => item.key === state.route);
  document.getElementById("page-title").textContent = domain?.title || "BuildML EDA Studio";
  document.getElementById("page-sub").textContent = domain?.short || "";
  const pdf = document.getElementById("pdf-download");
  if (pdf && pdf.tagName === "A") {
    pdf.href = `/api/export/pdf?view=${encodeURIComponent(state.route)}`;
  }
  const offlineHtml = document.getElementById("offline-html-download");
  if (offlineHtml && offlineHtml.tagName === "A") {
    offlineHtml.href = "/api/export/html";
  }
  renderNav();
  setMobileNavOpen(false);
  try {
    if (state.route === "academy") {
      await renderAcademy();
    } else if (state.route === "cockpit") {
      await renderCockpit();
    } else {
      await renderDomain(state.route);
    }
  } catch (error) {
    main().innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
  }
  hydrateIcons(document);
}

function currentRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return hash || "cockpit";
}

async function boot() {
  hydrateIcons(document);
  const saved = localStorage.getItem("buildml-eda-theme") || "light";
  setTheme(saved);

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    setTheme(next);
  });
  document.getElementById("drawer-close").addEventListener("click", closeConcept);
  document.getElementById("drawer-backdrop").addEventListener("click", closeConcept);
  document.getElementById("nav-toggle").addEventListener("click", () => {
    const open = !document.querySelector(".sidebar")?.classList.contains("open");
    setMobileNavOpen(open);
  });
  document.getElementById("mobile-nav-scrim")?.addEventListener("click", () => {
    setMobileNavOpen(false);
  });
  document.getElementById("severity-filter").addEventListener("change", (event) => {
    state.severity = event.target.value;
    navigate(state.route);
  });
  document.getElementById("section-filter").addEventListener("input", (event) => {
    state.sectionFilter = event.target.value;
    navigate(state.route);
  });

  let searchTimer;
  document.getElementById("global-search").addEventListener("input", (event) => {
    window.clearTimeout(searchTimer);
    const q = event.target.value.trim();
    searchTimer = window.setTimeout(async () => {
      if (!q) return;
      if (state.route === "academy") {
        await renderAcademy(q);
        return;
      }
      const results = await api(`/api/search?q=${encodeURIComponent(q)}`);
      toast(`${results.findings.length} findings · ${results.concepts.length} concepts`);
      if (results.domains?.[0]) {
        // soft-suggest first domain match without forcing navigation
      }
      main().insertAdjacentHTML(
        "afterbegin",
        `<section class="panel" id="search-panel">
          <div class="panel-head"><div><h2>Search</h2><p>Matches for “${escapeHtml(q)}”.</p></div></div>
          <div class="chip-row">
            ${(results.domains || []).map((d) => `<a class="chip" href="#/${escapeHtml(d.key)}">${escapeHtml(d.title)}</a>`).join("")}
            ${(results.concepts || []).slice(0, 8).map((c) => `<button type="button" class="chip" data-concept="${escapeHtml(c.key)}">${escapeHtml(c.title)}</button>`).join("")}
          </div>
        </section>`,
      );
      wireConceptChips(document.getElementById("search-panel"));
    }, 280);
  });

  window.addEventListener("hashchange", () => navigate(currentRoute()));
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement?.tagName !== "INPUT") {
      event.preventDefault();
      document.getElementById("global-search").focus();
    }
    if (event.key === "Escape") closeConcept();
  });

  try {
    state.meta = await api("/api/meta");
    await navigate(currentRoute());
  } catch (error) {
    main().innerHTML = `<div class="error-state">Failed to start studio: ${escapeHtml(error.message)}</div>`;
  }
}

boot();
