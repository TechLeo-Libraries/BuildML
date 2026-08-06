/**
 * BuildML EDA App: Industry readiness sheets.
 * Cockpit spine 01-08, Concept Academy, Readiness Gates, domain boards.
 * Gate marks are UI-only / in-memory (never persisted).
 *
 * Shared learn presentation: import from ./learn_ui.js (also wired in index.html).
 * Academy / Gates view modules should import the same primitives; do not fork markup.
 */
import { hydrateIcons, iconSvg } from "./icons.js";
import { renderAcademy as renderAcademyView } from "./academy_view.js";
import {
  callout,
  sectionScaffold,
  whatToChange,
  wireLearnUi,
} from "./learn_ui.js";
import {
  closeGateDrawer,
  renderGatesView,
  wireGateDrawerChrome,
} from "./gates_view.js";
import {
  closeCockpitDrawer,
  renderAssumptionsBody,
  renderLedgerBody,
  wireCockpitDrawerChrome,
  wireCockpitSheet,
} from "./cockpit_view.js";

const state = {
  meta: null,
  charts: null,
  chartsTheme: null,
  route: "cockpit",
  sort: { key: null, dir: "asc" },
  // Ephemeral only: cleared on refresh. Never written to the server or disk.
  gateSessionMarks: Object.create(null),
  gateFilter: "all",
  activeGateIdRef: { current: null },
  academyMode: "all",
  academyStages: [],
  academyQuery: "",
  sectionFilter: "",
};

const main = () => document.getElementById("main");
const offlineBundle = () => window.__BUILDML_OFFLINE__ || null;

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

async function api(path) {
  const offline = offlineBundle();
  if (offline) return offlineApi(path, offline);
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
  if (pathname === "/api/gates") return structuredClone(bundle.gates || bundle.domains?.gates?.gates);
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
    throw new Error("CSV/PDF export requires the live EDA App server.");
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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function themeTokens() {
  const dark = currentTheme() === "dark";
  return {
    dark,
    ink: dark ? "#E8EEF5" : "#1d1f20",
    muted: dark ? "#9AA8B8" : "#5d5d60",
    grid: dark ? "#2A3340" : "#d4d4d7",
    hoverBg: dark ? "#171C24" : "#f5f5f8",
    accent: dark ? "#94bce3" : "#5980a6",
    critical: dark ? "#F07178" : "#B91C1C",
    series: dark
      ? ["#94bce3", "#5980a6", "#F0A35A", "#b5d9fd", "#728fab", "#FB7185", "#98989b", "#FBBF24"]
      : ["#5980a6", "#2c455d", "#B45309", "#416180", "#728fab", "#9F1239", "#424244", "#A16207"],
    heatmap: dark
      ? [
          [0.0, "#94bce3"],
          [0.5, "#1D2430"],
          [1.0, "#5980a6"],
        ]
      : [
          [0.0, "#2c455d"],
          [0.5, "#f5f5f8"],
          [1.0, "#5980a6"],
        ],
  };
}

function chartLayoutOverrides() {
  const t = themeTokens();
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: 'Barlow, "Segoe UI", "Helvetica Neue", sans-serif',
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
    uniformtext: { minsize: 10, mode: "hide" },
    coloraxis: {
      colorbar: {
        tickfont: { color: t.ink },
        title: { font: { color: t.ink } },
      },
    },
    xaxis: {
      gridcolor: t.grid,
      zeroline: false,
      automargin: true,
      tickfont: { color: t.ink },
      title: { font: { color: t.ink } },
      linecolor: t.grid,
    },
    yaxis: {
      gridcolor: t.grid,
      zeroline: false,
      automargin: true,
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
    uniformtext: { ...(base.uniformtext || {}), ...overrides.uniformtext },
  };
  if (Array.isArray(base.annotations)) {
    const muted = overrides.meta.annotation_color;
    layout.annotations = base.annotations.map((item) => ({
      ...item,
      font: { ...(item.font || {}), color: muted },
    }));
  }
  for (const key of Object.keys(base)) {
    if (/^xaxis\d+$/.test(key)) layout[key] = { ...base[key], ...overrides.xaxis };
    if (/^yaxis\d+$/.test(key)) layout[key] = { ...base[key], ...overrides.yaxis };
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
    if (next.type === "bar") {
      next.constraintext = next.constraintext === "hide" ? "none" : next.constraintext || "none";
      next.cliponaxis = false;
      if (next.textposition === "outside" || !next.textposition) {
        next.textposition = "outside";
      }
    }
    if (!next.marker && (next.type === "bar" || next.type === "scatter")) {
      next.marker = { color: t.series[index % t.series.length] };
    }
    return next;
  });
}

async function renderChart(host, chartId) {
  host.setAttribute("data-chart-id", chartId);
  host.innerHTML = `<div class="empty" style="margin:0;padding:var(--space-4)">Loading chart…</div>`;
  try {
    const charts = await ensureCharts();
    const fig = charts[chartId];
    if (!fig) {
      host.innerHTML = `<div class="empty" style="margin:0">Chart unavailable.</div>`;
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

async function mountCharts(root) {
  const hosts = [...root.querySelectorAll("[data-mount-chart]")];
  await Promise.all(hosts.map((host) => renderChart(host, host.getAttribute("data-mount-chart"))));
}

function corners() {
  return `<i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>`;
}

function sevTag(row) {
  const label = escapeHtml(row.sev_label || row.severity || "info");
  if (row.is_blocking) return `<span class="tag tag-block">${label}</span>`;
  if (row.is_med) return `<span class="tag tag-accent">${label}</span>`;
  if (row.is_low) return `<span class="tag tag-outline">${label}</span>`;
  return `<span class="tag tag-neutral">${label}</span>`;
}

function conceptLink(key, label) {
  if (!key || key === "—") {
    return `<span class="om-slug text-muted">${escapeHtml(label || "—")}</span>`;
  }
  return `<button type="button" class="om-slug" data-concept="${escapeHtml(key)}" style="background:none;border:0;padding:0;cursor:pointer;color:var(--color-accent-700)">${escapeHtml(label || key)} →</button>`;
}

function updateChrome({ kicker, title, actionsHtml }) {
  const kickerEl = document.getElementById("sheet-kicker");
  const titleEl = document.getElementById("sheet-title");
  const actionsEl = document.getElementById("sheet-actions");
  if (kickerEl) kickerEl.textContent = kicker || "";
  if (titleEl) titleEl.textContent = title || "";
  if (actionsEl && actionsHtml != null) actionsEl.innerHTML = actionsHtml;
}

function primaryNavActions(active) {
  const offline = Boolean(offlineBundle());
  // App header: Offline HTML is the sole primary export. CSV/PDF stay on API
  // routes for automation; Static EDA keeps its own print/PDF path.
  const offlineHtml = offline
    ? `<span class="btn btn-primary blueprint" title="This file is already an offline snapshot">${corners()}Offline snapshot</span>`
    : `<a class="btn btn-primary blueprint" id="html-download" href="/api/export/html" title="Download offline HTML snapshot">${corners()}Offline HTML</a>`;

  if (active === "cockpit") {
    return `
      <a class="btn btn-ghost" href="#/gates">Readiness gates →</a>
      <a class="btn btn-ghost" href="#/academy">Concept academy →</a>
      ${offlineHtml}`;
  }
  if (active === "gates") {
    return `
      <a class="btn btn-ghost" href="#/cockpit">← Readiness sheet</a>
      <a class="btn btn-ghost" href="#/academy">Concept academy →</a>
      ${offlineHtml}`;
  }
  if (active === "academy") {
    return `
      <a class="btn btn-ghost" href="#/cockpit">← Readiness sheet</a>
      <a class="btn btn-ghost" href="#/gates">Readiness gates →</a>
      ${offlineHtml}`;
  }
  return `
    <a class="btn btn-ghost" href="#/cockpit">← Readiness sheet</a>
    <a class="btn btn-ghost" href="#/gates">Gates</a>
    <a class="btn btn-ghost" href="#/academy">Academy</a>
    ${offlineHtml}`;
}

function renderBoardNav() {
  const nav = document.getElementById("domain-nav");
  if (!nav) return;
  const primary = new Set(["cockpit", "gates", "academy"]);
  const domains = state.meta?.domains || [];
  const primaryLinks = domains.filter((d) => primary.has(d.key));
  const secondary = domains.filter((d) => !primary.has(d.key));
  nav.innerHTML = [
    ...primaryLinks.map((domain) => {
      const current = domain.key === state.route ? ' aria-current="page"' : "";
      return `<a href="#/${escapeHtml(domain.key)}"${current}>${escapeHtml(domain.title)}</a>`;
    }),
    secondary.length
      ? `<span class="om-mono om-kick" style="margin-left:var(--space-2)">Boards</span>`
      : "",
    ...secondary.map((domain) => {
      const current = domain.key === state.route ? ' aria-current="page"' : "";
      return `<a href="#/${escapeHtml(domain.key)}"${current}>${escapeHtml(domain.title)}</a>`;
    }),
  ].join("");
}

function figureCards(ids) {
  if (!ids?.length) return "";
  return `
    <div class="figure-grid">
      ${ids
        .map(
          (id, index) => `
        <figure class="fig-card blueprint">
          ${corners()}
          <figcaption>Fig. ${index + 1} — ${escapeHtml(id.replaceAll("_", " "))}</figcaption>
          <div class="chart-actions">
            <button type="button" class="btn btn-ghost om-mono" data-expand-chart="${escapeHtml(id)}" style="font-size:11px">Expand</button>
          </div>
          <div class="chart-host" data-mount-chart="${escapeHtml(id)}"></div>
        </figure>`,
        )
        .join("")}
    </div>`;
}

/* ── Cockpit (EDA Sheet - Cockpit) ─────────────────────────────── */

async function renderCockpit() {
  const data = await api("/api/cockpit");
  const sheet = data.sheet || {};
  const kpis = sheet.kpis || {};
  const register = sheet.register || [];
  const assumptions = sheet.assumptions || [];
  const ledger = sheet.ledger || [];
  const sequence = sheet.sequence || [];
  const domainBriefs = sheet.domain_briefs || [];
  const methods = sheet.methods || [];
  const degraded = sheet.degraded || [];
  const narrative = sheet.narrative || data.narrative || [];
  const meta = sheet.spine_meta || {};
  const adapt = sheet.adapt || data.adapt || {};
  const chartIds = sheet.chart_ids || data.chart_ids || [];
  const version = kpis.version || "";
  const engine = kpis.engine || data.overview?.engine || adapt.engine || "pandas";
  const sessionSentence =
    sheet.session_sentence || adapt.session_sentence || "";

  updateChrome({
    kicker: `BuildML ${version} · Exploratory data analysis · ${engine}`,
    title: "Command cockpit readiness sheet",
    actionsHtml: primaryNavActions("cockpit"),
  });

  const adaptChips = [
    adapt.target_column
      ? `target · ${adapt.target_column}`
      : "target · undeclared",
    adapt.task ? `task · ${adapt.task}` : null,
    adapt.n_columns != null ? `${adapt.n_columns} columns` : null,
    adapt.eligible_features
      ? `${adapt.eligible_features.length} eligible`
      : null,
    sheet.coverage?.ledger_items != null
      ? `${sheet.coverage.ledger_items} ledger numbers`
      : null,
    (adapt.skipped_analyzers || []).length
      ? `${adapt.skipped_analyzers.length} analyzers skipped/n-a`
      : null,
  ].filter(Boolean);

  const registerBody = `
        <div class="table-wrap">
        <table class="table table--fit table--register">
          <thead>
            <tr>
              <th class="col-sev">Sev</th>
              <th class="col-key">Key</th>
              <th class="col-detail">Detail</th>
              <th class="col-evidence">Evidence</th>
              <th class="col-concept">Concept</th>
            </tr>
          </thead>
          <tbody>
            ${
              register.length
                ? register
                    .map((row) => {
                      const cols = row.affected_columns || [];
                      const chips = cols.length
                        ? `<div class="col-chips">${cols
                            .slice(0, 8)
                            .map(
                              (c) =>
                                `<span class="col-chip om-mono" title="${escapeHtml(String(c))}">${escapeHtml(String(c))}</span>`,
                            )
                            .join("")}${
                            cols.length > 8
                              ? `<span class="col-chip om-mono text-muted">+${cols.length - 8}</span>`
                              : ""
                          }</div>`
                        : "";
                      return `
              <tr id="${escapeHtml(row.anchor || "")}" class="cockpit-register-row"
                  data-finding-key="${escapeHtml(row.key || "")}" tabindex="0" role="button"
                  title="Open finding teaching sidebar">
                <td class="col-sev">${sevTag(row)}</td>
                <td class="om-mono cell-wrap cell-wrap--key" style="font-size:12px">${escapeHtml(row.key)}</td>
                <td class="cell-wrap cell-wrap--prose">
                  ${escapeHtml(row.detail)}
                  ${(row.caveats || []).length
                    ? `<div class="sheet-note sheet-note--inline">${escapeHtml((row.caveats || []).join(" · "))}</div>`
                    : ""}
                </td>
                <td class="cell-wrap cell-wrap--evidence">
                  ${chips}
                  <div class="om-mono text-muted" style="font-size:11px">${escapeHtml(row.evidence)}</div>
                </td>
                <td class="cell-wrap cell-wrap--concept">${conceptLink(row.concept_key || row.concept, row.concept)}</td>
              </tr>`;
                    })
                    .join("")
                : `<tr><td colspan="5" class="text-muted">No findings raised for this frame.</td></tr>`
            }
          </tbody>
        </table>
        </div>
        ${
          narrative.length
            ? `<details class="sheet-details" style="margin-top:var(--space-3)"><summary class="om-mono om-kick">Narrative summary · ${narrative.length}</summary><ul class="sheet-bullets">${narrative
                .map((line) => `<li>${escapeHtml(line)}</li>`)
                .join("")}</ul></details>`
            : ""
        }`;

  const assumptionsBody = renderAssumptionsBody(
    assumptions,
    sheet.assumptions_purpose || {},
  );
  const ledgerBody = renderLedgerBody(
    ledger,
    sheet.ledger_purpose || {},
    sheet.ledger_glossary || {},
  );

  const sequenceBody = `
        <div class="table-wrap">
        <table class="table table--fit table--sequence">
          <thead>
            <tr>
              <th class="col-n">#</th>
              <th class="col-priority">Priority</th>
              <th class="col-action">Action</th>
              <th class="col-call">Call</th>
              <th class="col-based">Based on</th>
            </tr>
          </thead>
          <tbody>
            ${
              sequence.length
                ? sequence
                    .map(
                      (r) => `
              <tr>
                <td class="om-mono col-n" style="font-size:12px">${escapeHtml(r.n)}</td>
                <td class="om-mono cell-wrap" style="font-size:11px">${escapeHtml(r.when)}</td>
                <td class="cell-wrap cell-wrap--prose">
                  ${escapeHtml(r.title)}
                  ${r.rationale ? `<div class="sheet-note sheet-note--inline">${escapeHtml(r.rationale)}</div>` : ""}
                  ${(r.caveats || []).length
                    ? `<div class="sheet-note">${escapeHtml((r.caveats || []).join(" · "))}</div>`
                    : ""}
                </td>
                <td class="om-mono cell-wrap" style="font-size:11px">${escapeHtml(r.call)}</td>
                <td class="om-mono cell-wrap" style="font-size:11px">${escapeHtml(r.basis)}</td>
              </tr>`,
                    )
                    .join("")
                : `<tr><td colspan="5" class="text-muted">No recommendations produced.</td></tr>`
            }
          </tbody>
        </table>
        </div>
        ${whatToChange(sheet.what_to_change || [], { title: "What to change next" })}
        ${callout(
          "advanced",
          "Recommendations name Session operations; they do not execute them. Full-dataset descriptive EDA after a split summarises observed rows and is not train-fitted transform evidence.",
        )}`;

  const domainsBody = `
        <div class="domain-brief-grid">
          ${
            domainBriefs.length
              ? domainBriefs
                  .map(
                    (brief) => `
            <article class="domain-brief blueprint">
              ${corners()}
              <header class="domain-brief__head">
                <h5>${escapeHtml(brief.title || brief.key || "")}</h5>
                <a class="om-slug" href="#/${escapeHtml(brief.board || brief.key || "")}">Board →</a>
              </header>
              <p class="domain-brief__summary">${escapeHtml(brief.summary || "")}</p>
              <ul class="sheet-bullets">
                ${(brief.highlights || [])
                  .filter(Boolean)
                  .map((h) => `<li>${escapeHtml(h)}</li>`)
                  .join("")}
              </ul>
              ${(brief.metrics || []).length
                ? `<div class="domain-brief__metrics">${(brief.metrics || [])
                    .map(
                      (m) =>
                        `<div class="om-led" title="${escapeHtml(`${m.k} — ${m.v}`)}"><span class="om-led__key om-mono" title="${escapeHtml(m.k)}">${escapeHtml(m.k)}</span><span class="om-led__val om-mono" title="${escapeHtml(m.v)}">${escapeHtml(m.v)}</span></div>`,
                    )
                    .join("")}</div>`
                : ""}
              ${(brief.findings || []).length
                ? `<details class="sheet-details"><summary class="om-mono om-kick">Linked findings · ${(brief.findings || []).length}</summary><ul class="sheet-bullets">${(brief.findings || [])
                    .map((f) => `<li>${escapeHtml(f)}</li>`)
                    .join("")}</ul></details>`
                : ""}
            </article>`,
                  )
                  .join("")
              : `<p class="text-muted">No domain boards produced measurable briefs for this frame.</p>`
          }
        </div>`;

  const figuresBody = chartIds.length
    ? figureCards(chartIds)
    : `<p class="text-muted">No non-empty figures for this session — empty theater omitted.</p>`;

  const methodStatusCounts = methods.reduce(
    (acc, card) => {
      const s = card.status || "skipped";
      acc[s] = (acc[s] || 0) + 1;
      return acc;
    },
    {},
  );
  const methodsBody = `
        <div class="methods-legend" aria-label="Analyzer status counts">
          <span class="tag tag-accent">Ran · ${methodStatusCounts.ran || 0}</span>
          <span class="tag tag-outline">Skipped · ${methodStatusCounts.skipped || 0}</span>
          <span class="tag tag-neutral">Not applicable · ${methodStatusCounts.not_applicable || 0}</span>
        </div>
        <div class="methods-grid">
          ${
            methods.length
              ? methods
                  .map((card) => {
                    const status = card.status || "skipped";
                    const label =
                      status === "ran"
                        ? "Ran"
                        : status === "not_applicable"
                          ? "Not applicable"
                          : "Skipped";
                    const tagCls =
                      status === "ran"
                        ? "tag-accent"
                        : status === "not_applicable"
                          ? "tag-neutral"
                          : "tag-outline";
                    return `
            <article class="method-card blueprint" data-status="${escapeHtml(status)}">
              ${corners()}
              <header class="method-card__head">
                <h5>${escapeHtml(card.family || "")}</h5>
                <span class="tag ${tagCls}">${escapeHtml(label)}</span>
              </header>
              <p>${escapeHtml(card.summary || "")}</p>
              ${card.why ? `<p class="text-muted"><span class="assumption-card__label">Why</span> ${escapeHtml(card.why)}</p>` : ""}
              ${card.detail ? `<details class="sheet-details"><summary class="om-mono om-kick">Technical detail</summary><p class="om-mono" style="font-size:11.5px">${escapeHtml(card.detail)}</p></details>` : ""}
            </article>`;
                  })
                  .join("")
              : `<p class="text-muted">Methods catalog unavailable.</p>`
          }
        </div>
        ${callout(
          "advanced",
          "Associations describe co-occurrence, not causality. Empty analyzer sections are omitted from figures and ledger groups rather than filled with placeholders.",
          { title: "Caveats" },
        )}`;

  const degradedBody = `
        <div class="table-wrap">
        <table class="table table--fit table--degraded">
          <thead><tr><th class="col-analysis">Analysis</th><th class="col-reason">Reason</th></tr></thead>
          <tbody>
            ${
              degraded.length
                ? degraded
                    .map(
                      (row) => `
              <tr>
                <td class="om-mono cell-wrap" style="font-size:12px">${escapeHtml(row.analysis)}</td>
                <td class="cell-wrap cell-wrap--prose">${escapeHtml(row.reason)}</td>
              </tr>`,
                    )
                    .join("")
                : `<tr><td colspan="2" class="text-muted">No degraded or skipped analyses were recorded.</td></tr>`
            }
          </tbody>
        </table>
        </div>`;

  main().innerHTML = `
    <section class="kpi-strip blueprint">
      ${corners()}
      <div class="kpi">
        <div class="kpi__label">Readiness</div>
        <div class="kpi__value">${escapeHtml(kpis.readiness || data.readiness?.status || "—")}</div>
        <div class="kpi__note">${escapeHtml(kpis.readiness_note || "")}</div>
      </div>
      <div class="kpi">
        <div class="kpi__label">Scope</div>
        <div class="kpi__value om-mono">${escapeHtml(kpis.scope || "—")}</div>
        <div class="kpi__note">${escapeHtml(kpis.scope_note || "")}</div>
      </div>
      <div class="kpi">
        <div class="kpi__label">Completeness</div>
        <div class="kpi__value om-mono">${escapeHtml(kpis.completeness || "—")}</div>
        <div class="kpi__note">${escapeHtml(kpis.completeness_note || "")}</div>
      </div>
      <div class="kpi">
        <div class="kpi__label">Runtime</div>
        <div class="kpi__value om-mono">${escapeHtml(kpis.runtime || engine)}</div>
        <div class="kpi__note">${escapeHtml(kpis.runtime_note || "")}</div>
      </div>
    </section>

    <div class="adapt-strip" aria-label="Session-adaptive context">
      <div class="adapt-strip__label">This session</div>
      <div>${escapeHtml(sessionSentence)}</div>
      <div class="adapt-strip__chips">
        ${adaptChips.map((chip) => `<span class="adapt-chip">${escapeHtml(chip)}</span>`).join("")}
      </div>
    </div>

    ${(data.warnings || []).length
      ? callout(
          "warning",
          (data.warnings || []).map((w) => String(w)).join(" · "),
          { title: "Sampling / scope disclosures" },
        )
      : ""}

    ${sectionScaffold({ n: "01", title: "Findings register", meta: meta.register, bodyHtml: registerBody })}
    ${sectionScaffold({
      n: "02",
      title: (sheet.assumptions_purpose && sheet.assumptions_purpose.title) || "What each finding assumes",
      meta: meta.assumptions,
      bodyHtml: assumptionsBody,
      id: "cockpit-section-assumptions",
    })}
    ${sectionScaffold({
      n: "03",
      title: (sheet.ledger_purpose && sheet.ledger_purpose.title) || "Ledger — every computed number",
      meta: meta.ledger,
      bodyHtml: ledgerBody,
      id: "cockpit-section-ledger",
    })}
    ${sectionScaffold({ n: "04", title: "Recommended sequence", meta: meta.sequence, bodyHtml: sequenceBody })}
    ${sectionScaffold({ n: "05", title: "Domain briefs", meta: meta.domains, bodyHtml: domainsBody })}
    ${sectionScaffold({ n: "06", title: "Figures", meta: meta.figures, bodyHtml: figuresBody })}
    ${sectionScaffold({ n: "07", title: "Methods and limitations", meta: meta.methods, bodyHtml: methodsBody })}
    ${sectionScaffold({ n: "08", title: "Skipped and degraded analyses", meta: meta.degraded, bodyHtml: degradedBody })}
  `;
  await mountCharts(main());
  wireConceptChips(main());
  wireLearnUi(main());
  wireCockpitSheet(main(), sheet, {
    onCloseSiblings: () => {
      closeGateDrawer();
      closeConcept();
    },
  });
}

/* ── Gates (EDA Sheet - Readiness Gates) ───────────────────────── */

async function renderGates() {
  const data = await api("/api/gates");
  const version = state.meta?.session?.version || "";
  const engine = state.meta?.overview?.engine || "pandas";

  const toggleMark = (id) => {
    if (state.gateSessionMarks[id]) {
      delete state.gateSessionMarks[id];
      toast("Session mark cleared (still not saved anywhere)");
    } else {
      state.gateSessionMarks[id] = true;
      toast("Marked for this browser tab only — refresh clears it");
    }
    void renderGates();
  };

  renderGatesView(main(), data, {
    sessionMarks: state.gateSessionMarks,
    filter: state.gateFilter || "all",
    activeGateIdRef: state.activeGateIdRef,
    onFilter: (key) => {
      state.gateFilter = key;
      void renderGates();
    },
    onToggleMark: toggleMark,
    onOpenConcept: (key) => openConcept(key),
    onToast: toast,
    onChrome: ({ outstanding, total }) => {
      updateChrome({
        kicker: `${engine} ${version} · second pass · ${outstanding} of ${total} gates outstanding`,
        title: "Readiness gates",
        actionsHtml: primaryNavActions("gates"),
      });
    },
  });
  wireConceptChips(main());
}

/* ── Academy (EDA Sheet - Academy) ─────────────────────────────── */

async function renderAcademy(query) {
  if (typeof query === "string") state.academyQuery = query;
  await renderAcademyView({
    main,
    api,
    updateChrome,
    primaryNavActions,
    wireConceptChips,
    toast,
    state,
  });
}

/* ── Domain boards (secondary sheets) ──────────────────────────── */

function tableFromRows(rows, tableId) {
  if (!rows?.length) return `<div class="empty" style="margin:0">No tabular evidence in this section.</div>`;
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
  return `
    <div class="table-wrap">
      <table class="table table--fit" id="${tableId}">
        <thead><tr>${keys
          .map(
            (key) =>
              `<th scope="col" data-sort-key="${escapeHtml(key)}">${escapeHtml(key)}</th>`,
          )
          .join("")}</tr></thead>
        <tbody>${sorted
          .slice(0, 200)
          .map(
            (row) =>
              `<tr>${keys
                .map(
                  (key) =>
                    `<td class="cell-wrap">${escapeHtml(formatCell(row[key]))}</td>`,
                )
                .join("")}</tr>`,
          )
          .join("")}</tbody>
      </table>
    </div>
    <p class="sheet-note">Showing ${Math.min(sorted.length, 200)} of ${sorted.length} rows.</p>
  `;
}

function formatCell(value) {
  if (value == null) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(4);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function rowsFromSection(section) {
  if (!section) return [];
  if (Array.isArray(section)) return section.filter((item) => item && typeof item === "object");
  if (typeof section !== "object") return [];
  if (section.per_column && typeof section.per_column === "object" && !Array.isArray(section.per_column)) {
    return Object.entries(section.per_column).map(([column, stats]) => {
      if (stats && typeof stats === "object" && !Array.isArray(stats)) {
        const flat = { column };
        for (const [key, value] of Object.entries(stats)) {
          if (key === "iqr_bounds" && Array.isArray(value) && value.length === 2) {
            flat.iqr_lower = value[0];
            flat.iqr_upper = value[1];
          } else if (value == null || typeof value !== "object") {
            flat[key] = value;
          }
        }
        return flat;
      }
      return { column, value: stats };
    });
  }
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
          typeof v === "object" && v && !Array.isArray(v) ? { key: k, ...v } : { key: k, value: v },
        );
      }
    }
  }
  return Object.entries(section)
    .filter(([, v]) => typeof v !== "object" || v === null)
    .map(([key, value]) => ({ key, value }));
}

async function renderDomain(domainKey) {
  const data = await api(`/api/domains/${domainKey}`);
  const domain = data.domain || {};
  updateChrome({
    kicker: `BuildML · domain board · ${domain.key || domainKey}`,
    title: domain.title || domainKey,
    actionsHtml: primaryNavActions(domainKey),
  });

  const sections = data.sections || {};
  const tables = Object.entries(sections)
    .map(([name, payload], index) => {
      const rows = rowsFromSection(payload);
      const n = String(index + 1).padStart(2, "0");
      return `
        <div class="spine">
          <div class="spine__n">${n}</div>
          <section>
            <div class="spine__head"><h4>${escapeHtml(name)}</h4></div>
            ${tableFromRows(rows, `table-${name}`)}
          </section>
        </div>`;
    })
    .join("");

  main().innerHTML = `
    <p class="lead">${escapeHtml(domain.short || "")}</p>
    ${figureCards(data.chart_ids || [])}
    ${tables || `<div class="empty">No domain tables.</div>`}
  `;
  await mountCharts(main());
  wireConceptChips(main());
  wireTableSort(main());
}

/* ── Concept drawer ────────────────────────────────────────────── */

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
        .map(
          (item) =>
            `<button type="button" class="om-chip" data-concept="${escapeHtml(item.key)}">${escapeHtml(item.title)}</button>`,
        )
        .join("") || '<span class="text-muted">None listed.</span>'}</div>
    </div>
  `;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  wireConceptChips(document.getElementById("drawer-body"));
}

function closeConcept() {
  const drawer = document.getElementById("concept-drawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  document.getElementById("drawer-backdrop").hidden = true;
  document.body.style.overflow = "";
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

/* ── Navigation / boot ─────────────────────────────────────────── */

async function navigate(route) {
  state.route = route || "cockpit";
  renderBoardNav();
  try {
    if (state.route === "academy") {
      await renderAcademy();
    } else if (state.route === "gates") {
      await renderGates();
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
  const raw = window.location.hash.replace(/^#\/?/, "");
  // Legacy / accidental in-sheet anchors must never become domain routes.
  // (Previously bare ledger-* hashes were parsed as unknown domain boards.)
  if (!raw || raw === "cockpit") return "cockpit";
  if (
    raw.startsWith("ledger-") ||
    raw.startsWith("cockpit-ledger-") ||
    raw.startsWith("cockpit-section-") ||
    raw.startsWith("f-")
  ) {
    return "cockpit";
  }
  return raw;
}

async function boot() {
  hydrateIcons(document);
  // Industry sheets are light-first; keep a theme key for offline parity
  // without a chrome toggle. Never persist gate marks.
  document.documentElement.setAttribute("data-theme", "light");
  try {
    localStorage.setItem("buildml-eda-theme", "light");
  } catch {
    /* private mode */
  }

  document.getElementById("drawer-close")?.addEventListener("click", closeConcept);
  document.getElementById("drawer-backdrop")?.addEventListener("click", closeConcept);
  wireGateDrawerChrome();
  wireCockpitDrawerChrome();
  window.addEventListener("hashchange", () => navigate(currentRoute()));
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const cockpitDrawer = document.getElementById("cockpit-drawer");
    if (cockpitDrawer?.classList.contains("open")) {
      closeCockpitDrawer();
      return;
    }
    const gateDrawer = document.getElementById("gate-drawer");
    if (gateDrawer?.classList.contains("open")) {
      closeGateDrawer();
      state.activeGateIdRef.current = null;
      return;
    }
    closeConcept();
  });

  try {
    state.meta = await api("/api/meta");
    await navigate(currentRoute());
  } catch (error) {
    main().innerHTML = `<div class="error-state">Failed to start EDA App: ${escapeHtml(error.message)}</div>`;
  }
}

boot();
