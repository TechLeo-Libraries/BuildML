/**
 * BuildML Learn UI — shared presentation primitives for Academy, Gates, Cockpit.
 *
 * Academy / Gates agents should import from this module (do not fork markup):
 *
 *   import {
 *     callout, codeBlock, calcBlock, whatToChange, sectionScaffold, escapeHtml,
 *   } from "./learn_ui.js";
 *
 * Wire order in templates/index.html (module scripts):
 *   learn_ui.js  →  academy_view.js / gates_view.js / cockpit_view.js  →  app.js
 *
 * Callout types: tip | advanced | evidence | warning
 * All HTML helpers escape text content; pass pre-built HTML only via *Html options.
 */

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const CALLOUT_LABELS = {
  tip: "Beginner tip",
  beginner: "Beginner tip",
  advanced: "Advanced note",
  evidence: "Evidence",
  warning: "Warning",
};

/**
 * @param {"tip"|"beginner"|"advanced"|"evidence"|"warning"} type
 * @param {string} body
 * @param {{ title?: string, html?: boolean }} [opts]
 */
export function callout(type, body, opts = {}) {
  const kind = String(type || "tip").toLowerCase();
  const cls = kind === "beginner" ? "tip" : kind;
  const title = opts.title || CALLOUT_LABELS[kind] || CALLOUT_LABELS.tip;
  const content = opts.html ? String(body ?? "") : escapeHtml(body);
  return `
    <aside class="learn-callout learn-callout--${escapeHtml(cls)}" role="note">
      <div class="learn-callout__label om-mono">${escapeHtml(title)}</div>
      <div class="learn-callout__body">${content}</div>
    </aside>`;
}

/**
 * Copyable code / worked-example block.
 * @param {string} code
 * @param {{ label?: string, copyable?: boolean, lang?: string }} [opts]
 */
export function codeBlock(code, opts = {}) {
  const label = opts.label || "Worked example";
  const copyable = opts.copyable !== false;
  const lang = opts.lang ? ` data-lang="${escapeHtml(opts.lang)}"` : "";
  const text = String(code ?? "");
  const copyBtn = copyable
    ? `<button type="button" class="btn btn-ghost om-mono learn-code__copy" data-learn-copy style="font-size:10px;letter-spacing:.06em;text-transform:uppercase">Copy</button>`
    : "";
  return `
    <div class="learn-code" data-learn-code>
      <div class="learn-code__head">
        <span class="om-mono om-kick">${escapeHtml(label)}</span>
        ${copyBtn}
      </div>
      <pre class="learn-code__pre om-code"${lang}><code>${escapeHtml(text)}</code></pre>
    </div>`;
}

/**
 * Calculation / arithmetic-check block (power, strata, rows-per-feature, …).
 * @param {string} title
 * @param {Array<string|{label?:string, value?:string, text?:string}>} lines
 */
export function calcBlock(title, lines = []) {
  const rows = (lines || [])
    .map((line) => {
      if (typeof line === "string") {
        return `<li>${escapeHtml(line)}</li>`;
      }
      if (line && typeof line === "object") {
        if (line.label != null || line.value != null) {
          return `<li><span class="om-mono">${escapeHtml(line.label ?? "")}</span>
            <span class="om-mono learn-calc__value">${escapeHtml(line.value ?? "")}</span></li>`;
        }
        return `<li>${escapeHtml(line.text ?? "")}</li>`;
      }
      return "";
    })
    .join("");
  return `
    <div class="learn-calc">
      <div class="om-mono om-kick learn-calc__title">${escapeHtml(title || "Calculation")}</div>
      <ul class="learn-calc__list">${rows}</ul>
    </div>`;
}

/**
 * "What to change" checklist bound to live recommendations / gaps.
 * @param {Array<{change?:string, title?:string, why?:string, api?:string, call?:string}>} items
 * @param {{ title?: string }} [opts]
 */
export function whatToChange(items = [], opts = {}) {
  const title = opts.title || "What to change";
  if (!items?.length) {
    return `
      <div class="learn-change">
        <div class="om-mono om-kick">${escapeHtml(title)}</div>
        <p class="text-muted" style="margin:var(--space-2) 0 0;font-size:12.5px">Nothing forced by this session's numbers yet.</p>
      </div>`;
  }
  const rows = items
    .map((item) => {
      const change = item.change || item.title || "";
      const why = item.why || "";
      const api = item.api || item.call || "";
      return `
        <li class="learn-change__item">
          <div class="learn-change__action">${escapeHtml(change)}</div>
          ${why ? `<div class="learn-change__why">${escapeHtml(why)}</div>` : ""}
          ${api ? `<code class="learn-change__api om-mono">${escapeHtml(api)}</code>` : ""}
        </li>`;
    })
    .join("");
  return `
    <div class="learn-change">
      <div class="om-mono om-kick">${escapeHtml(title)}</div>
      <ol class="learn-change__list">${rows}</ol>
    </div>`;
}

/**
 * Numbered spine section scaffold (Industry sheet IA).
 * @param {{ n: string|number, title: string, meta?: string, bodyHtml: string, id?: string }} opts
 */
export function sectionScaffold(opts = {}) {
  const n = String(opts.n ?? "").padStart(2, "0").slice(-2);
  const title = opts.title || "";
  const meta = opts.meta
    ? `<span class="om-mono spine__meta">${escapeHtml(opts.meta)}</span>`
    : "";
  const idAttr = opts.id ? ` id="${escapeHtml(opts.id)}"` : "";
  return `
    <div class="spine"${idAttr}>
      <div class="spine__n">${escapeHtml(n)}</div>
      <section>
        <div class="spine__head">
          <h4>${escapeHtml(title)}</h4>
          ${meta}
        </div>
        ${opts.bodyHtml || ""}
      </section>
    </div>`;
}

/** Clipboard helper used by Academy / Gates views via ``BuildMLLearnUI``. */
export async function copyText(text) {
  const value = String(text ?? "");
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = value;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

/** Wire copy buttons inside a root (call after innerHTML assignment). */
export function wireLearnUi(root) {
  if (!root) return;
  root.querySelectorAll("[data-learn-copy]").forEach((btn) => {
    if (btn.dataset.learnBound) return;
    btn.dataset.learnBound = "1";
    btn.addEventListener("click", async () => {
      const host = btn.closest("[data-learn-code]");
      const code = host?.querySelector("code")?.textContent || "";
      try {
        await copyText(code);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        window.setTimeout(() => {
          btn.textContent = prev || "Copy";
        }, 1400);
      } catch {
        btn.textContent = "Copy failed";
      }
    });
  });
}

export { escapeHtml };

/** Global hook for offline shells / Academy view (non-import consumers). */
if (typeof window !== "undefined") {
  window.BuildMLLearnUI = {
    callout,
    codeBlock,
    calcBlock,
    whatToChange,
    sectionScaffold,
    wireLearnUi,
    copyText,
    escapeHtml,
  };
}
