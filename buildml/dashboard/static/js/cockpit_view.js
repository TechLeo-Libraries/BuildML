/**
 * Command Cockpit readiness sheet — sections 02/03 redesign + teaching sidebar.
 * Uses learn_ui.js primitives (same bar as Gates). Ledger jump chips never
 * navigate to /api/domains — they scroll in-sheet or open the cockpit drawer.
 */

import {
  callout,
  calcBlock,
  codeBlock,
  escapeHtml,
  whatToChange,
  wireLearnUi,
} from "./learn_ui.js";

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
  return `<ul class="gate-learn__list">${items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("")}</ul>`;
}

function detailsSection(title, bodyHtml, open = false) {
  if (!bodyHtml) return "";
  return `
    <details class="gate-learn__details" ${open ? "open" : ""}>
      <summary>${escapeHtml(title)}</summary>
      <div class="gate-learn__details-body">${bodyHtml}</div>
    </details>`;
}

function calculationHtml(calc) {
  if (!calc) return "";
  const lines = [
    { label: "formula", value: calc.formula || "" },
    ...Object.entries(calc.inputs || {}).map(([key, value]) => ({
      label: key,
      value: typeof value === "object" ? JSON.stringify(value) : String(value),
    })),
    { label: "result", value: calc.result || "" },
    calc.reading || "",
  ];
  return calcBlock(calc.label || "Calculation (this session)", lines);
}

function changeItems(worked) {
  return (worked.change_these || []).map((change) => ({ change }));
}

function flexibleItems(worked) {
  return (worked.flexible || []).map((change) => ({
    change,
    why: "Optional — adapt to your protocol or install extras.",
  }));
}

function kindLabel(kind) {
  if (kind === "ledger_group") return "Ledger group";
  if (kind === "ledger_metric") return "Ledger metric";
  if (kind === "assumption") return "Assumption footnote";
  if (kind === "finding") return "Findings register";
  return "Cockpit";
}

function flattenLedgerItems(group) {
  const items = [];
  for (const col of group.cols || []) {
    for (const it of col.items || []) {
      items.push({ k: String(it.k ?? ""), v: String(it.v ?? "") });
    }
  }
  return items;
}

function ledRowHtml(k, v, { groupKey, clickable = true } = {}) {
  const key = String(k ?? "");
  const val = String(v ?? "");
  const attrs = clickable
    ? ` role="button" tabindex="0" data-cockpit-open="ledger_metric" data-ledger-group="${escapeHtml(
        groupKey || "",
      )}" data-ledger-k="${escapeHtml(key)}" data-ledger-v="${escapeHtml(val)}"`
    : "";
  return `
    <div class="om-led cockpit-led"${attrs} title="${escapeHtml(`${key} — ${val}`)}">
      <span class="om-led__key om-mono" title="${escapeHtml(key)}">${escapeHtml(key)}</span>
      <span class="om-led__val om-mono" title="${escapeHtml(val)}">${escapeHtml(val)}</span>
    </div>`;
}

export function renderAssumptionsBody(assumptions, purpose = {}) {
  const themes = [];
  const seen = new Set();
  for (const a of assumptions || []) {
    const theme = String(a.theme || "general");
    if (!seen.has(theme)) {
      seen.add(theme);
      themes.push(theme);
    }
  }

  const purposeBlock = `
    <div class="cockpit-section-purpose">
      ${callout("tip", purpose.purpose || "", { title: "Why this section exists" })}
      ${
        purpose.how_to_use
          ? callout("evidence", purpose.how_to_use, { title: "How to use it" })
          : ""
      }
    </div>
    <div class="cockpit-toolbar" data-assumption-toolbar>
      <span class="om-kick">Theme</span>
      <button type="button" class="om-chip om-chip-on" data-assumption-filter="all">All ${
        assumptions?.length || 0
      }</button>
      ${themes
        .map(
          (t) =>
            `<button type="button" class="om-chip" data-assumption-filter="${escapeHtml(
              t,
            )}">${escapeHtml(t)}</button>`,
        )
        .join("")}
      <label class="cockpit-filter-label">
        <span class="om-kick">Filter</span>
        <input type="search" class="cockpit-filter-input" data-assumption-search
          placeholder="Filter footnotes" autocomplete="off" />
      </label>
    </div>`;

  if (!assumptions?.length) {
    return `${purposeBlock}<p class="text-muted">No assumption footnotes for this session.</p>`;
  }

  const cards = assumptions
    .map((a, index) => {
      const slug = a.slug || a.concept_key || `note-${index}`;
      const theme = a.theme || "general";
      const search = [slug, theme, a.means, a.matters, a.next, a.technical, a.evidence]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return `
      <details class="assumption-card assumption-card--collapsible" data-assumption-card
        data-theme="${escapeHtml(theme)}" data-search="${escapeHtml(search)}" ${
          index < 4 ? "open" : ""
        }>
        <summary class="assumption-card__summary">
          <span class="assumption-card__slug om-mono" title="${escapeHtml(slug)}">${escapeHtml(
            slug,
          )}</span>
          <span class="assumption-card__meta">
            ${theme ? `<span class="om-mono text-muted">${escapeHtml(theme)}</span>` : ""}
            <button type="button" class="om-chip" data-cockpit-open="assumption"
              data-assumption-key="${escapeHtml(slug)}">Learn →</button>
          </span>
        </summary>
        <div class="assumption-card__lanes">
          <div class="assumption-lane">
            <div class="assumption-card__label">Means</div>
            <p>${escapeHtml(a.means || "—")}</p>
          </div>
          <div class="assumption-lane">
            <div class="assumption-card__label">Matters</div>
            <p>${escapeHtml(a.matters || "—")}</p>
          </div>
        </div>
        ${
          a.next
            ? `<p><span class="assumption-card__label">Check next</span> ${escapeHtml(a.next)}</p>`
            : ""
        }
        <details class="sheet-details">
          <summary class="om-mono om-kick">Technical · evidence</summary>
          ${
            a.technical
              ? `<p class="assumption-card__prose text-muted"><span class="assumption-card__label">Technical</span> ${escapeHtml(
                  a.technical,
                )}</p>`
              : ""
          }
          ${
            a.evidence
              ? `<p class="om-mono assumption-card__evidence assumption-card__prose"><span class="assumption-card__label">Evidence</span> ${escapeHtml(
                  a.evidence,
                )}</p>`
              : ""
          }
        </details>
      </details>`;
    })
    .join("");

  return `
    ${purposeBlock}
    <div class="assumption-grid assumption-grid--deep assumption-grid--lanes">${cards}</div>`;
}

export function renderLedgerBody(ledger, purpose = {}, glossary = {}) {
  const purposeBlock = `
    <div class="cockpit-section-purpose">
      ${callout("tip", purpose.purpose || "", { title: "Why this section exists" })}
      ${
        purpose.how_to_use
          ? callout("evidence", purpose.how_to_use, { title: "How to use it" })
          : ""
      }
      ${callout(
        "warning",
        "Jump chips scroll inside the readiness sheet. They are not domain boards — clicking never calls /api/domains/ledger-*.",
        { title: "Navigation" },
      )}
    </div>`;

  if (!ledger?.length) {
    return `${purposeBlock}<p class="text-muted">Ledger unavailable.</p>`;
  }

  const totalItems = ledger.reduce((n, g) => n + (g.item_count || flattenLedgerItems(g).length), 0);

  const jump = `
    <div class="ledger-jump" aria-label="Ledger groups">
      ${ledger
        .map((g) => {
          const key = g.key || "";
          const gloss = glossary[key] || {};
          const tip = gloss.means || g.means || g.title || key;
          const count = g.item_count != null ? g.item_count : flattenLedgerItems(g).length;
          return `<button type="button" class="ledger-jump__chip om-mono"
            data-ledger-jump="${escapeHtml(key)}"
            title="${escapeHtml(tip)}">${escapeHtml(g.title)}${
              count != null ? ` · ${count}` : ""
            }</button>`;
        })
        .join("")}
    </div>`;

  const glossaryStrip = `
    <details class="sheet-details ledger-glossary">
      <summary class="om-mono om-kick">Ledger group glossary · what each chip means</summary>
      <dl class="ledger-glossary__dl">
        ${ledger
          .map((g) => {
            const key = g.key || "";
            const gloss = glossary[key] || {};
            const means = gloss.means || g.means || "";
            const why = gloss.why_on_sheet || g.why_on_sheet || "";
            return `<div class="ledger-glossary__row">
              <dt class="om-mono">${escapeHtml(g.title)}</dt>
              <dd>
                <div>${escapeHtml(means)}</div>
                ${why ? `<div class="text-muted" style="margin-top:4px">${escapeHtml(why)}</div>` : ""}
              </dd>
            </div>`;
          })
          .join("")}
      </dl>
    </details>`;

  const toolbar = `
    <div class="cockpit-toolbar" data-ledger-toolbar>
      <span class="om-mono" style="font-size:11px;color:var(--color-neutral-700)">${escapeHtml(
        String(totalItems),
      )} metrics · ${escapeHtml(String(ledger.length))} groups</span>
      <label class="cockpit-filter-label">
        <span class="om-kick">Filter</span>
        <input type="search" class="cockpit-filter-input" data-ledger-search
          placeholder="Filter metrics by name or value" autocomplete="off" />
      </label>
      <button type="button" class="om-chip" data-ledger-expand="all">Expand all</button>
      <button type="button" class="om-chip" data-ledger-expand="none">Collapse all</button>
    </div>`;

  const groups = ledger
    .map((group, index) => {
      const key = group.key || `g${index}`;
      const items = flattenLedgerItems(group);
      const count = group.item_count != null ? group.item_count : items.length;
      const means = group.means || glossary[key]?.means || "";
      const why = group.why_on_sheet || glossary[key]?.why_on_sheet || "";
      const search = [group.title, means, why, ...items.map((it) => `${it.k} ${it.v}`)]
        .join(" ")
        .toLowerCase();

      // Prefer up to 2 scannable columns; exclusions get a dedicated two-lane list.
      const mid = Math.ceil(items.length / 2) || 1;
      const left = items.slice(0, mid);
      const right = items.slice(mid);
      const isExclusion = key === "exclusions";

      const rowsHtml = isExclusion
        ? `<div class="ledger-group__cols ledger-group__cols--stack">
            <div>${items.map((it) => ledRowHtml(it.k, it.v, { groupKey: key })).join("")}</div>
          </div>`
        : `<div class="ledger-group__cols">
            <div>${left.map((it) => ledRowHtml(it.k, it.v, { groupKey: key })).join("")}</div>
            ${
              right.length
                ? `<div>${right
                    .map((it) => ledRowHtml(it.k, it.v, { groupKey: key }))
                    .join("")}</div>`
                : ""
            }
          </div>`;

      return `
      <details class="ledger-group cockpit-ledger-group" data-ledger-group-card
        data-ledger-key="${escapeHtml(key)}" data-search="${escapeHtml(search)}"
        id="cockpit-ledger-${escapeHtml(key)}" ${index < 3 ? "open" : ""}>
        <summary class="ledger-group__summary">
          <span class="ledger-group__title">
            ${escapeHtml(group.title)}
            <span class="text-muted">· ${escapeHtml(String(count))}</span>
          </span>
          <span class="ledger-group__actions">
            <button type="button" class="om-chip" data-cockpit-open="ledger_group"
              data-ledger-group="${escapeHtml(key)}">Learn →</button>
          </span>
        </summary>
        ${
          means
            ? `<p class="ledger-group__blurb">${escapeHtml(means)}${
                why ? ` <span class="text-muted">${escapeHtml(why)}</span>` : ""
              }</p>`
            : ""
        }
        ${rowsHtml}
      </details>`;
    })
    .join("");

  return `
    ${purposeBlock}
    ${jump}
    ${glossaryStrip}
    ${toolbar}
    <div class="ledger-grid ledger-grid--cards">${groups}</div>`;
}

export function renderRegisterEnhancements(root) {
  if (!root) return;
  root.querySelectorAll("tr[data-finding-key]").forEach((row) => {
    row.classList.add("cockpit-register-row");
    row.setAttribute("tabindex", "0");
    row.setAttribute("role", "button");
  });
}

function renderCockpitDrawerBody(teaching) {
  const levels = teaching.levels || {};
  const worked = teaching.worked_example || {};
  const kind = teaching.kind || "cockpit";
  const title =
    teaching.kind === "ledger_metric"
      ? `${teaching.metric_key} = ${teaching.metric_value}`
      : teaching.title || teaching.key || "Cockpit";

  const sampleItems = teaching.sample_items || [];
  const sampleHtml = sampleItems.length
    ? `<div class="cockpit-drawer__sample">${sampleItems
        .map(
          (it) =>
            `<div class="om-led"><span class="om-led__key om-mono" title="${escapeHtml(
              it.k,
            )}">${escapeHtml(it.k)}</span><span class="om-led__val om-mono" title="${escapeHtml(
              it.v,
            )}">${escapeHtml(it.v)}</span></div>`,
        )
        .join("")}</div>`
    : "";

  return `
    <div class="gate-drawer__sticky">
      <div class="gate-drawer__sticky-top">
        <div>
          <div class="om-mono om-kick">${escapeHtml(kindLabel(kind))} · ${escapeHtml(
            teaching.key || "",
          )}</div>
          <h2 class="gate-drawer__title">${escapeHtml(title)}</h2>
        </div>
        <button type="button" class="btn btn-ghost" data-cockpit-drawer-close aria-label="Close cockpit panel">Close</button>
      </div>
      <div class="chip-row">
        <span class="om-pill om-p-clear">${escapeHtml(kindLabel(kind))}</span>
        ${
          teaching.theme
            ? `<span class="om-mono" style="font-size:10.5px">${escapeHtml(teaching.theme)}</span>`
            : ""
        }
        ${
          teaching.severity
            ? `<span class="om-pill om-p-open">${escapeHtml(teaching.severity)}</span>`
            : ""
        }
      </div>
      ${callout(
        "warning",
        teaching.session_note ||
          "Readings stay in this browser tab only. BuildML never saves cockpit judgments.",
        { title: "Session-local only" },
      )}
    </div>

    <div class="gate-drawer__scroll">
      <section class="gate-learn__section">
        <h3>What it means</h3>
        ${paragraphsHtml(teaching.means || teaching.beginner || "")}
        ${callout("tip", levels.beginner || teaching.beginner || "", { title: "Beginner" })}
      </section>

      <section class="gate-learn__section">
        <h3>Why it matters</h3>
        ${paragraphsHtml(teaching.why_it_matters || teaching.why_on_sheet || "")}
      </section>

      <section class="gate-learn__section">
        <h3>Evidence from this report</h3>
        ${callout("evidence", teaching.evidence || "", { title: "From this report" })}
        ${sampleHtml}
      </section>

      ${
        teaching.calculation
          ? `<section class="gate-learn__section"><h3>Calculation (this session)</h3>${calculationHtml(
              teaching.calculation,
            )}</section>`
          : ""
      }

      ${detailsSection(
        "Beginner → advanced",
        `
          ${callout("tip", levels.beginner || "", { title: "Beginner" })}
          ${callout("evidence", levels.intermediate || "", { title: "Intermediate" })}
          ${callout("advanced", levels.advanced || "", { title: "Advanced" })}
        `,
        false,
      )}

      <section class="gate-learn__section">
        <h3>Worked BuildML example</h3>
        <p>${escapeHtml(worked.summary || "")}</p>
        ${codeBlock(worked.code || "", {
          label: "BuildML Session example",
          lang: "python",
          copyable: true,
        })}
        ${whatToChange(changeItems(worked), { title: "Change these" })}
        ${whatToChange(flexibleItems(worked), { title: "Flexible areas" })}
        ${
          worked.reading
            ? callout("advanced", worked.reading, { title: "Reading the example" })
            : ""
        }
      </section>

      <section class="gate-learn__section">
        <h3>Next checks</h3>
        ${listHtml(teaching.next_checks || []) || '<p class="text-muted">No follow-ups listed.</p>'}
      </section>
    </div>
  `;
}

export function closeCockpitDrawer() {
  const drawer = document.getElementById("cockpit-drawer");
  const backdrop = document.getElementById("cockpit-drawer-backdrop");
  if (!drawer) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  if (backdrop) backdrop.hidden = true;
  const gate = document.getElementById("gate-drawer");
  const concept = document.getElementById("concept-drawer");
  if (!gate?.classList.contains("open") && !concept?.classList.contains("open")) {
    document.body.style.overflow = "";
  }
}

export function openCockpitDrawer(teaching, opts = {}) {
  const drawer = document.getElementById("cockpit-drawer");
  const backdrop = document.getElementById("cockpit-drawer-backdrop");
  if (!drawer || !teaching) return;
  // Close sibling drawers so only one learning panel is open.
  opts.onCloseSiblings?.();
  drawer.innerHTML = renderCockpitDrawerBody(teaching);
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  if (backdrop) backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  wireLearnUi(drawer);
  drawer.querySelectorAll("[data-cockpit-drawer-close]").forEach((node) => {
    node.addEventListener("click", () => closeCockpitDrawer());
  });
}

function resolveTeaching(sheet, kind, dataset) {
  if (kind === "assumption") {
    const key = dataset.assumptionKey;
    const note = (sheet.assumptions || []).find(
      (a) => (a.slug || a.concept_key) === key,
    );
    return note?.teaching || null;
  }
  if (kind === "finding") {
    const key = dataset.findingKey;
    const row = (sheet.register || []).find((r) => r.key === key);
    return row?.teaching || null;
  }
  if (kind === "ledger_group") {
    const key = dataset.ledgerGroup;
    const group = (sheet.ledger || []).find((g) => g.key === key);
    return group?.teaching || null;
  }
  if (kind === "ledger_metric") {
    const key = dataset.ledgerGroup;
    const group = (sheet.ledger || []).find((g) => g.key === key);
    if (!group?.teaching) return null;
    const teaching = {
      ...group.teaching,
      kind: "ledger_metric",
      metric_key: dataset.ledgerK,
      metric_value: dataset.ledgerV,
      title: `${dataset.ledgerK} = ${dataset.ledgerV}`,
      beginner: `Metric “${dataset.ledgerK}” = ${dataset.ledgerV} inside “${group.title}”. ${
        group.teaching.means || ""
      }`,
      evidence: `From this report’s “${group.title}” group: ${dataset.ledgerK} → ${dataset.ledgerV}.`,
    };
    return teaching;
  }
  return null;
}

function applyAssumptionFilters(root) {
  const theme =
    root.querySelector("[data-assumption-filter].om-chip-on")?.getAttribute(
      "data-assumption-filter",
    ) || "all";
  const q = (
    root.querySelector("[data-assumption-search]")?.value || ""
  ).trim().toLowerCase();
  root.querySelectorAll("[data-assumption-card]").forEach((card) => {
    const themeOk = theme === "all" || card.getAttribute("data-theme") === theme;
    const search = card.getAttribute("data-search") || "";
    const qOk = !q || search.includes(q);
    card.hidden = !(themeOk && qOk);
  });
}

function applyLedgerFilters(root) {
  const q = (root.querySelector("[data-ledger-search]")?.value || "").trim().toLowerCase();
  root.querySelectorAll("[data-ledger-group-card]").forEach((card) => {
    const search = card.getAttribute("data-search") || "";
    const match = !q || search.includes(q);
    card.hidden = !match;
    if (match && q) card.open = true;
    card.querySelectorAll("[data-ledger-row], .cockpit-led").forEach((row) => {
      const text = (row.getAttribute("title") || row.textContent || "").toLowerCase();
      row.hidden = Boolean(q) && !text.includes(q);
    });
  });
}

export function wireCockpitSheet(root, sheet, opts = {}) {
  if (!root || !sheet) return;

  root.querySelectorAll("[data-assumption-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      root.querySelectorAll("[data-assumption-filter]").forEach((b) => b.classList.remove("om-chip-on"));
      btn.classList.add("om-chip-on");
      applyAssumptionFilters(root);
    });
  });
  root.querySelector("[data-assumption-search]")?.addEventListener("input", () => {
    applyAssumptionFilters(root);
  });

  root.querySelector("[data-ledger-search]")?.addEventListener("input", () => {
    applyLedgerFilters(root);
  });
  root.querySelectorAll("[data-ledger-expand]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-ledger-expand");
      root.querySelectorAll("[data-ledger-group-card]").forEach((card) => {
        if (!card.hidden) card.open = mode === "all";
      });
    });
  });

  // Jump chips: scroll only — never change location.hash to ledger-*.
  root.querySelectorAll("[data-ledger-jump]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const key = btn.getAttribute("data-ledger-jump");
      const target = root.querySelector(`#cockpit-ledger-${CSS.escape(key || "")}`);
      if (target) {
        target.open = true;
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        target.classList.add("ledger-group--flash");
        window.setTimeout(() => target.classList.remove("ledger-group--flash"), 1200);
      }
      // Also open teaching for the group (Gates-quality deep dive).
      const group = (sheet.ledger || []).find((g) => g.key === key);
      if (group?.teaching) openCockpitDrawer(group.teaching, opts);
    });
  });

  const openFromDataset = (node) => {
    const kind = node.getAttribute("data-cockpit-open");
    const teaching = resolveTeaching(sheet, kind, {
      assumptionKey: node.getAttribute("data-assumption-key"),
      findingKey: node.getAttribute("data-finding-key"),
      ledgerGroup: node.getAttribute("data-ledger-group"),
      ledgerK: node.getAttribute("data-ledger-k"),
      ledgerV: node.getAttribute("data-ledger-v"),
    });
    if (teaching) openCockpitDrawer(teaching, opts);
  };

  root.querySelectorAll("[data-cockpit-open]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      openFromDataset(node);
    });
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openFromDataset(node);
      }
    });
  });

  // Prevent Learn buttons inside <summary> from toggling details only.
  root.querySelectorAll("summary .om-chip[data-cockpit-open]").forEach((btn) => {
    btn.addEventListener("click", (event) => event.preventDefault());
  });

  root.querySelectorAll("tr[data-finding-key]").forEach((row) => {
    const open = () => {
      const teaching = resolveTeaching(sheet, "finding", {
        findingKey: row.getAttribute("data-finding-key"),
      });
      if (teaching) openCockpitDrawer(teaching, opts);
    };
    row.addEventListener("click", (event) => {
      if (event.target.closest("[data-concept], a, button")) return;
      open();
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
}

export function wireCockpitDrawerChrome(opts = {}) {
  document.getElementById("cockpit-drawer-backdrop")?.addEventListener("click", () => {
    closeCockpitDrawer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const drawer = document.getElementById("cockpit-drawer");
    if (drawer?.classList.contains("open")) {
      closeCockpitDrawer();
      opts.onEscape?.();
    }
  });
}
