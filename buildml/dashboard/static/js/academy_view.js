/**
 * Concept Academy learning-hub view (Industry redesign + full CONCEPT_NOTES).
 *
 * Performance at ~200+ lessons:
 * - Stages collapse by default (open when filtered / jumped / toggled)
 * - Entries render as compact rows; full pedagogy expands on demand
 * - Search/filter recompute without mounting every worked example into the DOM
 *
 * Shared learn_ui primitives (do not fork markup for callout / code / calc / change).
 */

import {
  calcBlock,
  callout,
  codeBlock,
  escapeHtml,
  whatToChange,
  wireLearnUi,
} from "./learn_ui.js";

const CHIP_PREVIEW = 18;

function corners() {
  return `<i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>`;
}

function bullets(items, tag = "ul") {
  const list = (items || []).filter(Boolean);
  if (!list.length) return `<p class="text-muted" style="margin:0;font-size:12.5px">—</p>`;
  return `<${tag}>${list.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</${tag}>`;
}

function proseBlocks(lines) {
  return (lines || [])
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join("");
}

function section(title, bodyHtml, { open = false, key = "" } = {}) {
  if (!bodyHtml) return "";
  return `
    <details class="academy-disclose" data-academy-section="${escapeHtml(key)}" ${open ? "open" : ""}>
      <summary><span class="om-kick">${escapeHtml(title)}</span></summary>
      <div class="academy-disclose__body">${bodyHtml}</div>
    </details>`;
}

function calcSectionHtml(calc, item) {
  const formula = calc.formula || item.formula || "";
  const walkthrough = calc.walkthrough || item.calculation || "";
  if (!formula && !walkthrough) return "";
  const lines = [];
  if (formula) lines.push({ label: "formula", value: formula });
  if (walkthrough) {
    String(walkthrough)
      .split(/\n+/)
      .map((part) => part.trim())
      .filter(Boolean)
      .forEach((part) => lines.push(part));
  }
  return calcBlock("Calculation (this session)", lines);
}

function changeItems(raw) {
  return (raw || [])
    .filter(Boolean)
    .map((item) => (typeof item === "string" ? { change: item } : item));
}

function renderEntryFull(item, stage) {
  const sections = item.sections || {};
  const what = sections.what_it_means || item.prose || [item.summary];
  const technical = sections.technical_depth || [];
  const why = sections.why_it_matters || item.why || item.why_it_matters || [];
  const calc = sections.calculation || {};
  const worked = sections.worked_example || {};
  const evidence = sections.evidence || {};
  const read = sections.how_to_read || item.read || [];
  const pitfalls = sections.pitfalls || item.pitfalls || [];
  const decide = sections.decide || item.decide || "";
  const whatToChangeRaw = worked.what_to_change || item.what_to_change || [];
  const code = worked.code || item.example || "";
  const beginnerTip =
    (Array.isArray(what) && what[0]) ||
    item.summary ||
    "Read the plain-language block first, then open Calculation and the worked example.";
  const findingChips = (evidence.findings || [])
    .map(
      (f) =>
        `<span class="om-chip om-chip-on" title="${escapeHtml(f.severity || "")}">${escapeHtml(f.label || f.key)}</span>`,
    )
    .join("");

  return `
    <article class="academy-entry om-entry" id="${escapeHtml(item.key)}" data-academy-entry="${escapeHtml(item.key)}" data-expanded="1">
      <div class="academy-entry__rail ${item.cited ? "" : "ref"}"></div>
      <div class="academy-entry__main">
        <div class="academy-entry__head">
          <h3>${escapeHtml(item.title)}</h3>
          <span class="om-mono academy-stage-tag">stage ${escapeHtml(stage.n)}</span>
          ${
            item.readiness_path
              ? `<span class="om-chip om-chip-off">readiness</span>`
              : item.catalog
                ? `<span class="om-chip om-chip-off">catalog</span>`
                : `<span class="om-chip om-chip-off">lesson</span>`
          }
          ${
            item.cited
              ? `<button type="button" class="om-chip om-chip-on" data-concept="${escapeHtml(item.concept_key || item.key)}">cited ×${escapeHtml(item.cite_count)}</button>`
              : `<span class="om-chip om-chip-off">reference — no finding cited it</span>`
          }
          <button type="button" class="btn btn-ghost om-mono" data-academy-collapse="${escapeHtml(item.key)}" style="font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;margin-left:auto">Collapse</button>
        </div>

        <div class="academy-kicker-block">
          <span class="om-kick">What it means</span>
          ${proseBlocks(what)}
        </div>
        ${callout("tip", beginnerTip, { title: "Beginner tip" })}

        <p class="academy-session">
          <span class="om-mono">In this session / on this dataset</span> — ${escapeHtml(evidence.session || item.session || "")}
        </p>
        ${findingChips ? `<div class="chip-row" style="margin-top:var(--space-2)">${findingChips}</div>` : ""}

        ${
          decide
            ? `<div class="academy-decide"><span class="om-kick" style="display:block;margin-bottom:2px">The decision it forces</span>${escapeHtml(decide)}</div>`
            : ""
        }

        <div class="academy-disclose-stack">
          ${section("Why it matters", bullets(why), { key: "why" })}
          ${section(
            "Technical depth",
            `${proseBlocks(technical)}${
              technical.length
                ? callout("advanced", technical[technical.length - 1], {
                    title: "Advanced note",
                  })
                : ""
            }`,
            { key: "tech" },
          )}
          ${section(
            "Calculation",
            calcSectionHtml(calc, item),
            { key: "calc", open: Boolean(calc.walkthrough || item.calculation) },
          )}
          ${section("How to read it", bullets(read, "ol"), { key: "read", open: true })}
          ${section("Pitfalls", bullets(pitfalls), { key: "pitfalls" })}
          ${section(
            "What to change for your data",
            whatToChange(changeItems(whatToChangeRaw), {
              title: "What to change for your data",
            }),
            { key: "change" },
          )}
        </div>

        <div class="chip-row" style="margin-top:var(--space-4)">
          <button type="button" class="om-chip" data-concept="${escapeHtml(item.concept_key || item.key)}">Open full concept note</button>
        </div>
      </div>

      <aside class="academy-side">
        <div class="om-kick" style="margin-bottom:var(--space-2)">Worked BuildML example</div>
        ${codeBlock(code, { label: "BuildML · Python", lang: "python" })}
        ${
          whatToChangeRaw.length
            ? `<div class="om-kick" style="margin:var(--space-4) 0 var(--space-2)">Flexible bits</div>${whatToChange(
                changeItems(whatToChangeRaw),
                { title: "Flexible bits" },
              )}`
            : ""
        }
        <div class="om-kick" style="margin:var(--space-4) 0 var(--space-2)">Pitfalls</div>
        ${bullets(pitfalls)}
      </aside>
    </article>`;
}

function renderEntryCompact(item, stage) {
  const summary = item.summary || (item.prose && item.prose[0]) || "";
  const session = item.session || (item.sections && item.sections.evidence && item.sections.evidence.session) || "";
  return `
    <article class="academy-entry academy-entry--compact om-entry" id="${escapeHtml(item.key)}" data-academy-entry="${escapeHtml(item.key)}">
      <div class="academy-entry__rail ${item.cited ? "" : "ref"}"></div>
      <div class="academy-entry__main" style="grid-column:2 / -1">
        <div class="academy-entry__head">
          <h3 style="font-size:16px">${escapeHtml(item.title)}</h3>
          <span class="om-mono academy-stage-tag">stage ${escapeHtml(stage.n)}</span>
          ${item.cited ? `<span class="om-chip om-chip-on">cited ×${escapeHtml(item.cite_count)}</span>` : `<span class="om-chip om-chip-off">reference</span>`}
          ${item.readiness_path ? `<span class="om-chip om-chip-off">readiness</span>` : ""}
          <button type="button" class="btn btn-secondary om-mono" data-academy-expand="${escapeHtml(item.key)}" style="font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;margin-left:auto">Open lesson</button>
        </div>
        <p style="margin:0;font-size:12.5px;line-height:1.45;color:var(--color-neutral-800)">${escapeHtml(summary)}</p>
        <p class="academy-session" style="margin-top:var(--space-2)">
          <span class="om-mono">In this session</span> — ${escapeHtml(session)}
        </p>
      </div>
    </article>`;
}

function renderEntry(item, stage, openKeys) {
  if (openKeys.has(item.key)) return renderEntryFull(item, stage);
  return renderEntryCompact(item, stage);
}

function passesFilters(item, { mode, picked, needle }) {
  if (mode === "cited" && !item.cited) return false;
  if (mode === "ref" && item.cited) return false;
  if (mode === "readiness" && !item.readiness_path) return false;
  if (mode === "domain" && item.stage !== 6) return false;
  if (mode === "catalog" && !item.catalog) return false;
  if (picked.length && !picked.includes(item.stage)) return false;
  if (!needle) return true;
  const blob = (item.search || [
    item.key,
    item.title,
    item.summary,
    item.session,
    item.example,
    item.calculation,
    ...(item.prose || []),
    ...(item.read || []),
    ...(item.pitfalls || []),
    ...(item.what_to_change || []),
    ...(item.tags || []),
  ].join(" ")).toLowerCase();
  const terms = needle.split(/\s+/).filter(Boolean);
  return terms.every((term) => blob.includes(term));
}

function chipRowHtml(chips, stageKey, expandedStages) {
  const list = chips || [];
  const expanded = expandedStages.has(stageKey);
  const shown = expanded ? list : list.slice(0, CHIP_PREVIEW);
  const rest = list.length - shown.length;
  return `
    ${shown
      .map(
        (ch) =>
          `<button type="button" class="om-chip ${ch.cited ? "om-chip-on" : "om-chip-off"}" data-academy-jump="${escapeHtml(ch.key)}">${escapeHtml(ch.slug)}${ch.cited ? `<span class="om-mono" style="font-size:9px;opacity:.75">×${escapeHtml(ch.count)}</span>` : ""}</button>`,
      )
      .join("")}
    ${
      rest > 0
        ? `<button type="button" class="om-chip om-chip-off" data-academy-chips-more="${escapeHtml(stageKey)}">+${rest} more</button>`
        : ""
    }
    ${
      expanded && list.length > CHIP_PREVIEW
        ? `<button type="button" class="om-chip om-chip-off" data-academy-chips-less="${escapeHtml(stageKey)}">show fewer</button>`
        : ""
    }`;
}

/**
 * Render the Concept Academy board.
 * @param {object} deps
 */
export async function renderAcademy(deps) {
  const {
    main,
    api,
    updateChrome,
    primaryNavActions,
    wireConceptChips,
    state,
  } = deps;

  if (!state.academyOpenKeys) state.academyOpenKeys = [];
  if (!state.academyOpenStages) state.academyOpenStages = [];
  if (!state.academyChipExpand) state.academyChipExpand = [];

  const q = state.academyQuery || "";
  const data = await api("/api/domains/academy");
  const mode = state.academyMode || "all";
  const picked = state.academyStages || [];
  const needle = q.trim().toLowerCase();
  const engine = state.meta?.overview?.engine || "pandas";
  const version = state.meta?.session?.version || "";
  const adapt = data.adaptivity || {};
  const openKeys = new Set(state.academyOpenKeys || []);
  const openStages = new Set(state.academyOpenStages || []);
  const chipExpand = new Set(state.academyChipExpand || []);

  updateChrome({
    kicker: `${engine} ${version} · learning hub · ${data.concept_count ?? 0} lessons · ${data.catalog_count ?? "—"} catalog concepts · ${data.cited_count ?? 0} cited${
      adapt.target ? ` · target ${adapt.target}` : ""
    }${adapt.task ? ` · ${adapt.task}` : ""}`,
    title: "Concept academy",
    actionsHtml: primaryNavActions("academy"),
  });

  const filterOpts = { mode, picked, needle };
  const stages = (data.stages || [])
    .map((stage) => ({
      ...stage,
      entries: (stage.entries || []).filter((item) => passesFilters(item, filterOpts)),
    }))
    .filter((stage) => stage.entries.length);

  // Auto-open stages when searching or when a stage filter is active.
  const autoOpenAllStages = Boolean(needle) || picked.length > 0 || mode === "cited";
  const shown = stages.reduce((n, s) => n + s.entries.length, 0);
  const modeBtn = (key, label, on) =>
    `<button type="button" class="btn ${on ? "btn-primary" : "btn-secondary"}" data-academy-mode="${escapeHtml(key)}" style="font-size:11px;letter-spacing:.06em;text-transform:uppercase">${escapeHtml(label)}</button>`;

  const adaptLine = adapt.rows
    ? `Adaptive to this session: ${adapt.rows} rows × ${adapt.columns ?? "?"} cols` +
      (adapt.target ? ` · target “${adapt.target}”` : " · no target") +
      (adapt.task ? ` · ${adapt.task}` : "") +
      (adapt.has_mi ? " · MI available" : "") +
      (adapt.has_vif ? " · VIF available" : "")
    : data.curriculum_note || "";

  const readinessN = data.readiness_count ?? "—";
  const catalogN = data.catalog_count ?? "—";
  const domainN = (data.stages || []).find((s) => s.key === 6)?.chips?.length ?? 0;

  main().innerHTML = `
    <div class="academy-toolbar">
      <div class="academy-toolbar__row">
        <input class="input om-mono" id="academy-search" type="search" placeholder="Search all lessons — a statistic, a pitfall, a column name, a domain" value="${escapeHtml(q)}" style="font-size:12.5px;width:100%" />
        <div class="academy-toolbar__modes">
          ${modeBtn("all", `All ${data.concept_count ?? 0}`, mode === "all")}
          ${modeBtn("readiness", `Readiness ${readinessN}`, mode === "readiness")}
          ${modeBtn("catalog", `Catalog ${catalogN}`, mode === "catalog")}
          ${modeBtn("domain", `Domain ${domainN}`, mode === "domain")}
          ${modeBtn("cited", `Cited ${data.cited_count ?? 0}`, mode === "cited")}
          ${modeBtn("ref", `Reference ${(data.concept_count ?? 0) - (data.cited_count ?? 0)}`, mode === "ref")}
        </div>
        <div style="display:flex;gap:var(--space-3);align-items:baseline">
          <span class="om-mono" style="font-size:11px;color:var(--color-neutral-700);white-space:nowrap">${
            shown === data.concept_count
              ? `all ${data.concept_count} lessons`
              : `showing ${shown} of ${data.concept_count}`
          }</span>
          <button type="button" class="btn btn-ghost om-mono" id="academy-reset" style="font-size:11px;letter-spacing:.06em;text-transform:uppercase">Reset</button>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:var(--space-4);align-items:center">
        <div class="academy-toolbar__stages">
          <span class="om-kick">Stage</span>
          ${(data.stages || [])
            .map((stage) => {
              const on = picked.includes(stage.key);
              return `<button type="button" class="om-chip ${on ? "om-chip-on" : "om-chip-off"}" data-academy-stage="${escapeHtml(stage.key)}">${escapeHtml(stage.n)} ${escapeHtml(stage.label)}</button>`;
            })
            .join("")}
        </div>
        <div class="academy-legend">
          <span style="display:flex;gap:6px;align-items:center"><span class="om-chip om-chip-on">cited</span><span class="om-mono" style="font-size:10.5px;color:var(--color-neutral-700)">finding on this session rests on it</span></span>
          <span style="display:flex;gap:6px;align-items:center"><span class="om-chip om-chip-off">reference</span><span class="om-mono" style="font-size:10.5px;color:var(--color-neutral-700)">taught, not triggered here</span></span>
        </div>
      </div>
      <p class="academy-adapt om-mono">${escapeHtml(adaptLine)}</p>
    </div>

    <section class="academy-index blueprint">
      ${corners()}
      <div class="academy-index__head">
        <span class="om-mono om-kick" style="letter-spacing:.14em">Contents</span>
        <span class="om-mono" style="font-size:10.5px;color:var(--color-neutral-700)">every chip is a full lesson — filled = cited, outlined = reference · click to open</span>
      </div>
      ${(data.stages || [])
        .map(
          (stage) => `
        <div class="academy-index__row">
          <div>
            <div style="display:flex;align-items:baseline;gap:var(--space-2)">
              <span class="om-mono" style="font-size:26px;line-height:1;color:var(--color-accent);letter-spacing:-0.02em">${escapeHtml(stage.n)}</span>
              <span style="font-family:var(--font-heading);font-size:17px;letter-spacing:.04em;text-transform:uppercase">${escapeHtml(stage.label)}</span>
            </div>
            <div class="om-mono" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--color-neutral-700);margin-top:4px">${escapeHtml(stage.count_label)}</div>
            <div style="font-size:11.5px;line-height:1.45;color:var(--color-neutral-700);margin-top:6px;max-width:26ch">${escapeHtml(stage.blurb)}</div>
          </div>
          <div class="academy-index__chips">
            ${chipRowHtml(stage.chips, stage.key, chipExpand)}
          </div>
        </div>`,
        )
        .join("")}
    </section>

    ${
      stages.length
        ? stages
            .map((stage) => {
              const stageOpen =
                autoOpenAllStages ||
                openStages.has(stage.key) ||
                stage.entries.some((e) => openKeys.has(e.key));
              return `
      <details class="spine academy-stage" data-academy-stage-block="${escapeHtml(stage.key)}" ${stageOpen ? "open" : ""}>
        <summary class="spine__summary">
          <div class="spine__n">${escapeHtml(stage.n)}</div>
          <div class="spine__head" style="flex:1">
            <h4>${escapeHtml(stage.label)} — ${escapeHtml(stage.blurb)}</h4>
            <span class="om-mono" style="font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--color-neutral-700)">${stage.entries.length} lessons · click to ${stageOpen ? "collapse" : "expand"}</span>
          </div>
        </summary>
        <section>
          ${stage.entries.map((item) => renderEntry(item, stage, openKeys)).join("")}
        </section>
      </details>`;
            })
            .join("")
        : `<div class="empty">No concept matches the current filters.<div style="margin-top:var(--space-3)"><button type="button" class="btn btn-secondary om-mono" id="academy-reset-empty" style="font-size:11px;letter-spacing:.06em;text-transform:uppercase">Clear filters</button></div></div>`
    }
  `;

  const host = main();
  const input = document.getElementById("academy-search");
  let timer;
  input?.addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      state.academyQuery = input.value.trim();
      void renderAcademy(deps);
    }, 220);
  });

  host.querySelectorAll("[data-academy-mode]").forEach((node) => {
    node.addEventListener("click", () => {
      state.academyMode = node.getAttribute("data-academy-mode") || "all";
      void renderAcademy(deps);
    });
  });

  host.querySelectorAll("[data-academy-stage]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = Number(node.getAttribute("data-academy-stage"));
      if (Number.isNaN(key)) return;
      if (state.academyStages.includes(key)) {
        state.academyStages = state.academyStages.filter((item) => item !== key);
      } else {
        state.academyStages = [...state.academyStages, key];
      }
      void renderAcademy(deps);
    });
  });

  host.querySelectorAll("[data-academy-stage-block]").forEach((node) => {
    node.addEventListener("toggle", () => {
      const key = Number(node.getAttribute("data-academy-stage-block"));
      if (Number.isNaN(key)) return;
      const set = new Set(state.academyOpenStages || []);
      if (node.open) set.add(key);
      else set.delete(key);
      state.academyOpenStages = [...set];
    });
  });

  host.querySelectorAll("[data-academy-expand]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = node.getAttribute("data-academy-expand");
      if (!key) return;
      const set = new Set(state.academyOpenKeys || []);
      set.add(key);
      state.academyOpenKeys = [...set];
      void renderAcademy(deps).then(() => {
        const el = document.getElementById(key);
        el?.scrollIntoView({ block: "start", behavior: "smooth" });
      });
    });
  });

  host.querySelectorAll("[data-academy-collapse]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = node.getAttribute("data-academy-collapse");
      state.academyOpenKeys = (state.academyOpenKeys || []).filter((item) => item !== key);
      void renderAcademy(deps);
    });
  });

  host.querySelectorAll("[data-academy-chips-more]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = Number(node.getAttribute("data-academy-chips-more"));
      const set = new Set(state.academyChipExpand || []);
      set.add(key);
      state.academyChipExpand = [...set];
      void renderAcademy(deps);
    });
  });

  host.querySelectorAll("[data-academy-chips-less]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = Number(node.getAttribute("data-academy-chips-less"));
      state.academyChipExpand = (state.academyChipExpand || []).filter((item) => item !== key);
      void renderAcademy(deps);
    });
  });

  const reset = () => {
    state.academyMode = "all";
    state.academyStages = [];
    state.academyQuery = "";
    state.academyOpenKeys = [];
    state.academyOpenStages = [];
    state.academyChipExpand = [];
    void renderAcademy(deps);
  };
  document.getElementById("academy-reset")?.addEventListener("click", reset);
  document.getElementById("academy-reset-empty")?.addEventListener("click", reset);

  host.querySelectorAll("[data-academy-jump]").forEach((node) => {
    node.addEventListener("click", () => {
      const slug = node.getAttribute("data-academy-jump");
      if (!slug) return;
      const set = new Set(state.academyOpenKeys || []);
      set.add(slug);
      state.academyOpenKeys = [...set];
      state.academyMode = "all";
      state.academyStages = [];
      state.academyQuery = "";
      const jump = () => {
        const el = document.getElementById(slug);
        if (!el) return;
        window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 110 });
        el.animate(
          [{ background: "var(--color-accent-100)" }, { background: "transparent" }],
          { duration: 1400, easing: "ease-out" },
        );
      };
      void renderAcademy(deps).then(() => {
        setTimeout(jump, 0);
        setTimeout(jump, 140);
      });
    });
  });

  wireLearnUi(host);
  wireConceptChips(host);
}
