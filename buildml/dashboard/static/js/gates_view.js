/**
 * Readiness Gates sheet + deep-dive learning sidebar.
 * Uses learn_ui.js primitives (callout, codeBlock, calcBlock, whatToChange).
 * Gate marks remain in-memory only (passed via opts.sessionMarks).
 */

import {
  callout,
  calcBlock,
  codeBlock,
  escapeHtml,
  whatToChange,
  wireLearnUi,
} from "./learn_ui.js";

function corners() {
  return `<i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>`;
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
  return `<ul class="gate-learn__list">${items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("")}</ul>`;
}

function gatePillClass(status) {
  if (status === "session") return "om-p-session";
  if (status === "clear") return "om-p-clear";
  if (status === "open") return "om-p-open";
  if (status === "human") return "om-p-human";
  return "om-p-na";
}

function effectiveStatus(row, sessionMarks) {
  if (sessionMarks?.[row.id]) return "session";
  return row.status;
}

function statusLabel(row, sessionMarks) {
  if (sessionMarks?.[row.id]) return "marked for this session";
  return row.status_label || row.status;
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

function renderGateDrawerBody(row, sessionMarks) {
  const teaching = row.teaching || {};
  const levels = teaching.levels || {};
  const worked = teaching.worked_example || {};
  const meanings = teaching.status_meanings || {};
  const eff = effectiveStatus(row, sessionMarks);
  const marked = Boolean(sessionMarks?.[row.id]);

  return `
    <div class="gate-drawer__sticky">
      <div class="gate-drawer__sticky-top">
        <div>
          <div class="om-mono om-kick">Gate ${escapeHtml(row.id)} · ${escapeHtml(row.concept || "")}</div>
          <h2 class="gate-drawer__title">${escapeHtml(row.question)}</h2>
        </div>
        <button type="button" class="btn btn-ghost" data-gate-drawer-close aria-label="Close gate panel">Close</button>
      </div>
      <div class="chip-row">
        <span class="om-pill ${gatePillClass(eff)}">${escapeHtml(statusLabel(row, sessionMarks))}</span>
        ${
          row.concept_key
            ? `<button type="button" class="om-chip" data-concept="${escapeHtml(row.concept_key)}">Academy · ${escapeHtml(row.concept)}</button>`
            : `<span class="om-mono" style="font-size:10.5px">${escapeHtml(row.concept || "")}</span>`
        }
        ${
          row.session_mark_eligible
            ? `<button type="button" class="om-chip ${marked ? "om-chip-on" : ""}" data-gate-mark="${escapeHtml(row.id)}">${
                marked ? "Clear session mark" : "Mark for this session"
              }</button>`
            : ""
        }
      </div>
      ${callout(
        "warning",
        teaching.session_mark_note ||
          "Marks stay in this browser tab only and are discarded on refresh. BuildML never saves gate judgments.",
        { title: "Session-local only" },
      )}
    </div>

    <div class="gate-drawer__scroll">
      <section class="gate-learn__section">
        <h3>What this gate asks</h3>
        ${paragraphsHtml(teaching.beginner || row.question)}
        ${callout("tip", levels.beginner || teaching.beginner || "", { title: "Beginner" })}
      </section>

      <section class="gate-learn__section">
        <h3>Why it matters before modeling</h3>
        ${paragraphsHtml(teaching.why_before_modeling || "")}
      </section>

      <section class="gate-learn__section">
        <h3>How this status was derived</h3>
        ${callout("evidence", teaching.how_derived || row.evidence || "", {
          title: "From this report",
        })}
        <div class="gate-closes"><span class="om-kick" style="display:block;margin-bottom:2px">Closes when</span>${escapeHtml(
          teaching.closes_when || row.closes || "",
        )}</div>
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

      ${detailsSection(
        "What open / clear / human / N/A mean",
        `
          <dl class="gate-status-dl">
            <dt>Clear</dt><dd>${escapeHtml(meanings.clear || "")}</dd>
            <dt>Open</dt><dd>${escapeHtml(meanings.open || "")}</dd>
            <dt>Needs human judgment</dt><dd>${escapeHtml(meanings.human || "")}</dd>
            <dt>N/A</dt><dd>${escapeHtml(meanings.na || "")}</dd>
            <dt>Session mark</dt><dd>${escapeHtml(meanings.session_mark || teaching.session_mark_note || "")}</dd>
          </dl>
          <p class="gate-learn__reading">Current meaning for this gate: ${escapeHtml(
            teaching.status_meaning || statusLabel(row, sessionMarks),
          )}</p>
        `,
        false,
      )}

      <section class="gate-learn__section">
        <h3>Next checks</h3>
        ${listHtml(teaching.next_checks || []) || '<p class="text-muted">No follow-ups listed.</p>'}
      </section>

      ${
        (row.findings || []).length
          ? `<section class="gate-learn__section"><h3>Cited findings</h3>${listHtml(
              (row.findings || []).map((f) => f.label || f.key),
            )}</section>`
          : ""
      }
    </div>
  `;
}

export function closeGateDrawer() {
  const drawer = document.getElementById("gate-drawer");
  const backdrop = document.getElementById("gate-drawer-backdrop");
  if (!drawer) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  if (backdrop) backdrop.hidden = true;
  const concept = document.getElementById("concept-drawer");
  if (!concept?.classList.contains("open")) {
    document.body.style.overflow = "";
  }
}

export function openGateDrawer(row, opts = {}) {
  const drawer = document.getElementById("gate-drawer");
  const backdrop = document.getElementById("gate-drawer-backdrop");
  if (!drawer || !row) return;
  const sessionMarks = opts.sessionMarks || Object.create(null);
  drawer.innerHTML = renderGateDrawerBody(row, sessionMarks);
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  if (backdrop) backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  opts.activeGateIdRef && (opts.activeGateIdRef.current = row.id);

  wireLearnUi(drawer);

  drawer.querySelectorAll("[data-gate-drawer-close]").forEach((node) => {
    node.addEventListener("click", () => closeGateDrawer());
  });
  drawer.querySelectorAll("[data-gate-mark]").forEach((node) => {
    node.addEventListener("click", () => {
      const id = node.getAttribute("data-gate-mark");
      if (id) opts.onToggleMark?.(id);
    });
  });
  drawer.querySelectorAll("[data-concept]").forEach((node) => {
    node.addEventListener("click", () => {
      const key = node.getAttribute("data-concept");
      if (key) opts.onOpenConcept?.(key);
    });
  });
}

/**
 * Render the Gates board into `root` and wire interactions.
 */
export function renderGatesView(root, data, opts = {}) {
  const sessionMarks = opts.sessionMarks || Object.create(null);
  const filter = opts.filter || "all";
  const counts = data.counts || {};
  const outstanding = (counts.open || 0) + (counts.human || 0);
  const rowsById = new Map();

  for (const row of data.rows || []) {
    rowsById.set(row.id, row);
  }

  const groups = (data.groups || [])
    .map((group) => {
      const rows = (group.rows || []).filter((row) => {
        rowsById.set(row.id, row);
        if (filter === "all") return true;
        if (filter === "session") return Boolean(sessionMarks[row.id]);
        return row.status === filter;
      });
      return { ...group, rows };
    })
    .filter((group) => group.rows.length);

  const chip = (key, label, on) =>
    `<button type="button" class="om-chip ${on ? "om-chip-on" : ""}" data-gate-filter="${escapeHtml(key)}">${escapeHtml(label)}</button>`;

  const showing =
    filter === "all"
      ? `all ${counts.total ?? 0} gates`
      : `${groups.reduce((n, g) => n + g.rows.length, 0)} of ${counts.total ?? 0} gates`;

  root.innerHTML = `
    <p class="lead">The cockpit reports what this frame contains and the academy explains why it matters. This sheet is the second pass: every question that must be settled before the frame is modelled, with the status the session's own numbers support — and, deliberately, the questions no dataset can answer for you. Click a gate to open the learning sidebar.</p>
    <div class="gate-notice">${escapeHtml(
      data.ephemeral_notice ||
        "Human marks stay in this browser tab only and are discarded on refresh.",
    )}</div>

    <section class="tally-strip blueprint">
      ${corners()}
      <div class="kpi">
        <div class="kpi__value">${escapeHtml(counts.clear)}</div>
        <div class="om-kick" style="margin-top:var(--space-2)">settled by the frame</div>
        <div class="kpi__note" style="margin-top:4px">${escapeHtml(data.settled_pct)} of ${escapeHtml(
          counts.answerable,
        )} gates this dataset can answer on its own.</div>
      </div>
      <div class="kpi">
        <div class="kpi__value">${escapeHtml(counts.open)}</div>
        <div class="om-kick" style="margin-top:var(--space-2)">open and measurable</div>
        <div class="kpi__note" style="margin-top:4px">Each one names the number that would close it.</div>
      </div>
      <div class="kpi">
        <div class="kpi__value">${escapeHtml(counts.human)}</div>
        <div class="om-kick" style="margin-top:var(--space-2)">needs a human judgment</div>
        <div class="kpi__note" style="margin-top:4px">No dataset answers these; a person must write the answer down.</div>
      </div>
      <div class="kpi">
        <div class="kpi__value">${escapeHtml(counts.na)}</div>
        <div class="om-kick" style="margin-top:var(--space-2)">not applicable here</div>
        <div class="kpi__note" style="margin-top:4px">Kept visible so the next frame does not inherit the silence.</div>
      </div>
    </section>

    <div class="gate-filters">
      <div class="gate-filters__chips">
        <span class="om-kick">Show</span>
        ${chip("all", `All ${counts.total ?? 0}`, filter === "all")}
        ${chip("open", `Open ${counts.open ?? 0}`, filter === "open")}
        ${chip("human", `Needs a judgment ${counts.human ?? 0}`, filter === "human")}
        ${chip("clear", `Settled ${counts.clear ?? 0}`, filter === "clear")}
        ${chip("na", `Not applicable ${counts.na ?? 0}`, filter === "na")}
        ${chip("session", "This session", filter === "session")}
      </div>
      <span class="om-mono" style="font-size:11px;color:var(--color-neutral-700)">${escapeHtml(showing)}</span>
    </div>

    ${
      groups.length
        ? groups
            .map(
              (group) => `
      <div class="spine">
        <div class="spine__n">${escapeHtml(group.n)}</div>
        <section>
          <div class="spine__head">
            <h4>${escapeHtml(group.label)} — ${escapeHtml(group.blurb)}</h4>
            <span class="om-mono" style="font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--color-neutral-700)">${escapeHtml(
              group.count_label,
            )}</span>
          </div>
          ${group.rows
            .map((row) => {
              const eff = effectiveStatus(row, sessionMarks);
              const conceptKey = row.concept_key || row.concept;
              const marked = Boolean(sessionMarks[row.id]);
              return `
              <article class="gate-card" data-gate-open="${escapeHtml(row.id)}" tabindex="0" role="button" aria-label="Open gate ${escapeHtml(
                row.id,
              )} learning panel">
                <div class="gate-rail ${escapeHtml(eff)}"></div>
                <div class="gate-id">${escapeHtml(row.id)}</div>
                <div>
                  <div class="gate-card__question">${escapeHtml(row.question)}</div>
                  <div class="chip-row">
                    <span class="om-pill ${gatePillClass(eff)}">${escapeHtml(statusLabel(row, sessionMarks))}</span>
                    ${
                      conceptKey
                        ? `<button type="button" class="om-chip" data-concept="${escapeHtml(conceptKey)}">${escapeHtml(
                            row.concept,
                          )} →</button>`
                        : ""
                    }
                    ${(row.findings || [])
                      .map(
                        (f) =>
                          `<span class="om-mono" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--color-neutral-700)">finding ${escapeHtml(
                            f.label || f.key,
                          )}</span>`,
                      )
                      .join("")}
                  </div>
                  ${
                    row.session_mark_eligible
                      ? `<div class="gate-mark">
                          <button type="button" class="om-chip ${marked ? "om-chip-on" : ""}" data-gate-mark="${escapeHtml(
                            row.id,
                          )}">
                            ${marked ? "Clear session mark" : "Mark for this session"}
                          </button>
                          <span class="om-mono" style="font-size:10.5px;color:var(--color-neutral-700)">Local toggle only — lost on refresh</span>
                        </div>`
                      : ""
                  }
                  <button type="button" class="gate-card__learn" data-gate-open="${escapeHtml(row.id)}">Learn in sidebar →</button>
                </div>
                <div>
                  <div class="gate-card__evidence">${escapeHtml(row.evidence)}</div>
                  <div class="gate-closes"><span class="om-kick" style="display:block;margin-bottom:2px">Closes when</span>${escapeHtml(
                    row.closes,
                  )}</div>
                </div>
              </article>`;
            })
            .join("")}
        </section>
      </div>`,
            )
            .join("")
        : `<div class="empty">No gate has that status in this session.</div>`
    }
  `;

  opts.onChrome?.({ outstanding, total: counts.total ?? 0 });

  root.querySelectorAll("[data-gate-filter]").forEach((node) => {
    node.addEventListener("click", () => {
      opts.onFilter?.(node.getAttribute("data-gate-filter") || "all");
    });
  });

  root.querySelectorAll("[data-gate-mark]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      const id = node.getAttribute("data-gate-mark");
      if (id) opts.onToggleMark?.(id);
    });
  });

  const openFrom = (id) => {
    const row = rowsById.get(id);
    if (row) openGateDrawer(row, opts);
  };

  root.querySelectorAll("[data-gate-open]").forEach((node) => {
    node.addEventListener("click", (event) => {
      if (event.target.closest("[data-concept], [data-gate-mark]")) return;
      const id = node.getAttribute("data-gate-open");
      if (id) openFrom(id);
    });
    if (node.matches("article.gate-card")) {
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          const id = node.getAttribute("data-gate-open");
          if (id) openFrom(id);
        }
      });
    }
  });

  root.querySelectorAll("[data-concept]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      const key = node.getAttribute("data-concept");
      if (key) opts.onOpenConcept?.(key);
    });
  });

  const drawer = document.getElementById("gate-drawer");
  const activeId = opts.activeGateIdRef?.current;
  if (drawer?.classList.contains("open") && activeId) {
    const row = rowsById.get(activeId);
    if (row) openGateDrawer(row, opts);
  }

  return { rowsById, outstanding };
}

export function wireGateDrawerChrome() {
  document.getElementById("gate-drawer-backdrop")?.addEventListener("click", () => {
    closeGateDrawer();
  });
}
