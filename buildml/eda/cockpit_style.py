# ruff: noqa: E501
"""Industry design tokens and interactive shell for BUILDML STATIC EDA.

Inlined into the research HTML export so the report stays offline-capable: no
CDN, no Google Fonts fetch, no sidecar stylesheet. Font stacks name Barlow /
Barlow Condensed when present on the machine and fall back to condensed system
sans faces that preserve the blueprint density.
"""

from __future__ import annotations

from buildml.eda.industry_tokens import INDUSTRY_ROOT_CSS

COCKPIT_CSS = (
    INDUSTRY_ROOT_CSS
    + """
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; overflow-x: clip; }
body {
  margin: 0;
  background: var(--color-bg);
  color: var(--color-text);
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.55;
  font-weight: 400;
}
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-heading);
  font-weight: var(--font-heading-weight);
  line-height: 1.12;
  letter-spacing: -0.015em;
  margin: 0 0 var(--space-2);
}
h1 { font-size: 42px; text-transform: uppercase; letter-spacing: -0.01em; }
h4 { font-size: 20px; letter-spacing: .06em; text-transform: uppercase; }
p { margin: 0 0 var(--space-3); }
a { color: var(--color-accent-700); text-decoration: none; text-underline-offset: 3px; }
a:hover { color: var(--color-accent); text-decoration: underline; }
img { display: block; max-width: 100%; }
figure { margin: 0; }
.text-muted { color: color-mix(in srgb, var(--color-text) 55%, transparent); }
:focus { outline: none; }
:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
::selection { background: color-mix(in srgb, var(--color-accent) 30%, transparent); }

.bml-skip-link {
  position: absolute; left: 1rem; top: -5rem; z-index: 20;
  padding: .55rem .9rem; color: var(--color-bg); background: var(--color-accent-800);
}
.bml-skip-link:focus { top: 1rem; }

.om-shell {
  min-height: 100vh;
  padding: var(--space-6) var(--space-8) var(--space-8);
  max-width: var(--bml-max);
  margin: 0 auto;
}
.om-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  column-gap: clamp(2rem, 6vw, 4.5rem);
  row-gap: var(--space-6);
  align-items: end;
  border-bottom: 2px solid var(--color-text);
  padding-bottom: var(--space-4);
}
.om-header__title {
  display: grid;
  gap: var(--space-3);
  min-width: 0;
  padding-right: var(--space-4);
}
.om-header__title h1 {
  margin: 0;
  max-width: 28ch;
}
.om-kicker {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--color-accent-700);
  margin: 0;
}
.om-tools {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  align-items: center;
  justify-content: flex-end;
  margin-left: var(--space-4);
  padding-left: var(--space-4);
  border-left: 1px solid var(--color-divider);
  min-height: 2.75rem;
}
.om-mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.om-slug {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: .04em;
  white-space: nowrap;
  color: var(--color-accent-700);
}
.om-led {
  display: grid;
  grid-template-columns: minmax(0, 42%) minmax(0, 58%);
  column-gap: var(--space-3);
  align-items: baseline;
  padding: 3px 0;
  border-bottom: 1px dotted var(--color-divider);
  min-width: 0;
  width: 100%;
}
.om-led__key,
.om-led > span:first-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}
.om-led__val,
.om-led > span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}
.om-spine {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr);
  gap: var(--space-6);
  margin-top: var(--space-8);
}
.om-spine__n {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: .1em;
  color: var(--color-accent-700);
  padding-top: 4px;
}
.om-section-title {
  margin: 0;
  letter-spacing: .06em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--color-divider);
  padding-bottom: var(--space-2);
}
.om-kpi {
  margin-top: var(--space-8);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  background: var(--color-divider);
}
.om-kpi__cell { background: var(--color-bg); padding: var(--space-4); }
.om-kpi__label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: color-mix(in srgb, var(--color-text) 55%, transparent);
}
.om-kpi__value {
  font-family: var(--font-heading);
  font-size: 26px;
  letter-spacing: .02em;
  text-transform: uppercase;
  margin-top: 2px;
  color: var(--color-accent-800);
}
.om-kpi__value.om-mono { font-family: var(--font-mono); text-transform: none; }
.om-kpi__note {
  font-family: var(--font-mono);
  font-size: 11px;
  color: color-mix(in srgb, var(--color-text) 55%, transparent);
}
.om-panel { margin-top: var(--space-4); }
.om-panel__tools {
  display: grid;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
  padding-bottom: var(--space-3);
  border-bottom: 1px dashed var(--color-divider);
}
.om-panel__filter {
  display: grid;
  gap: 4px;
  max-width: 36rem;
  font-size: 11px;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--color-neutral-600);
}
.om-panel-search {
  width: 100%;
  min-height: 34px;
  padding: 6px 10px;
  font: inherit;
  font-size: 13px;
  text-transform: none;
  letter-spacing: normal;
  color: var(--color-text);
  background: var(--color-surface);
  border: 1px solid var(--color-divider);
}
.om-panel__count {
  margin: 0;
  font-size: 11px;
  color: var(--color-neutral-600);
}
.om-jump {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}
.om-jump__chip {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: .04em;
  color: var(--color-accent-800);
  background: var(--color-accent-100);
  border: 1px solid var(--color-accent-200);
  text-decoration: none;
}
.om-jump__chip:hover {
  background: var(--color-accent-200);
  text-decoration: none;
}
.om-group {
  margin: 0 0 var(--space-4);
  border-top: 1px solid var(--color-divider);
}
.om-group__summary {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-2);
  padding: var(--space-3) 0;
  cursor: pointer;
  list-style: none;
  background: color-mix(in srgb, var(--color-bg) 92%, transparent);
  backdrop-filter: blur(2px);
}
.om-group__summary::-webkit-details-marker { display: none; }
.om-group__title {
  font-family: var(--font-heading);
  font-size: 18px;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.om-group__meta {
  font-size: 11px;
  color: var(--color-neutral-600);
}
.om-assumption-groups { display: grid; gap: var(--space-2); }
.om-assumptions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-4) var(--space-6);
  margin: 0 0 var(--space-2);
}
.om-assumption-card {
  border-bottom: 1px dotted var(--color-divider);
  padding-bottom: var(--space-2);
}
.om-assumption-card__summary {
  display: grid;
  gap: var(--space-2);
  cursor: pointer;
  list-style: none;
}
.om-assumption-card__summary::-webkit-details-marker { display: none; }
.om-assumption-card__top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
}
.om-assumption-card__id {
  color: var(--color-accent-700);
  font-size: 12px;
}
.om-assumption-card__blurb {
  font-size: 13px;
  line-height: 1.45;
  color: var(--color-text);
}
.om-assumption-card__key {
  font-size: 11px;
}
.om-assumption-card__body {
  display: grid;
  gap: var(--space-2);
  margin-top: var(--space-3);
  padding-top: var(--space-2);
  border-top: 1px dashed var(--color-divider);
}
.om-assumption-card__body p,
.om-assumption p { margin: 0; font-size: 13px; line-height: 1.55; }
.om-assumption__label {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--color-accent-700);
  margin-right: var(--space-2);
}
.om-assumption__tech,
.om-assumption__evidence {
  color: color-mix(in srgb, var(--color-text) 72%, transparent);
}
.om-assumption__tech,
.om-assumption__evidence,
.om-assumption-card__body p,
.om-assumption-card__blurb {
  white-space: normal;
  overflow: visible;
  text-overflow: unset;
  overflow-wrap: anywhere;
  word-break: break-word;
  max-width: 100%;
}
.om-ledger {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: var(--space-4) var(--space-6);
  align-items: start;
}
.om-ledger-group {
  border-top: 1px solid var(--color-divider);
  min-width: 0;
}
.om-ledger__rows { display: grid; gap: 0; }
.om-methods { margin-top: var(--space-4); }
.om-methods__legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin-bottom: var(--space-4);
}
.om-methods__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: var(--space-4);
}
.om-method {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  background: var(--color-bg);
  min-height: 100%;
}
.om-method__head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-divider);
}
.om-method__family {
  margin: 0;
  font-family: var(--font-heading);
  font-size: 18px;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.om-method__summary {
  margin: 0;
  font-size: 13px;
  line-height: 1.55;
}
.om-method__why {
  margin: 0;
  font-size: 12px;
  color: color-mix(in srgb, var(--color-text) 72%, transparent);
}
.om-method__detail {
  margin: 0;
  font-size: 12px;
}
.om-method__detail summary {
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--color-accent-700);
}
.om-method__detail p {
  margin: var(--space-2) 0 0;
  font-size: 11px;
  line-height: 1.5;
  color: color-mix(in srgb, var(--color-text) 72%, transparent);
}
.om-callout {
  margin-top: var(--space-6);
  padding: var(--space-4);
  border: 1px solid var(--color-divider);
  background: color-mix(in srgb, var(--color-accent-100) 55%, var(--color-bg));
}
.om-callout__label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--color-accent-800);
  margin-bottom: var(--space-2);
}
.om-callout ul {
  margin: 0;
  padding-left: 1.1rem;
  font-size: 13px;
  line-height: 1.55;
}
.om-callout li + li { margin-top: var(--space-1); }
.om-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: var(--space-6);
  margin-top: var(--space-4);
}
.om-figure { margin: var(--space-4) 0 0; padding: var(--space-4); overflow-x: auto; }
.om-figure figcaption {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: .1em;
  text-transform: uppercase;
  margin: 0 0 var(--space-3);
  color: var(--color-accent-700);
}
/* Three-lane bar: category | track | value. Value lane is max-content so
   labels never wrap into the next row (the FIG. 5.2 collision class). */
.om-bar-row {
  display: grid;
  grid-template-columns: minmax(4.5rem, 28%) minmax(3rem, 1fr) minmax(6.75rem, max-content);
  column-gap: 8px;
  row-gap: 0;
  align-items: center;
  min-height: 20px;
  margin: 0;
  overflow: hidden;
}
.om-bar-track {
  position: relative;
  height: 11px;
  min-width: 0;
  border-left: 1px solid var(--color-text);
}
.om-bar-fill {
  position: absolute; left: 0; top: 0; bottom: 0;
  max-width: 100%;
  background: var(--color-accent-200);
  border: 1.5px solid var(--color-accent-700);
  box-sizing: border-box;
}
.om-bar-fill.is-hot {
  background: var(--color-accent-800);
  border-color: var(--color-accent-800);
}
.om-cap {
  font-size: 9px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: right;
  min-width: 0;
}
.om-bar-val {
  font-family: var(--font-mono);
  font-size: 9px;
  line-height: 1.2;
  color: var(--color-accent-800);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  text-align: right;
  justify-self: end;
  min-width: 0;
}
.om-profile {
  height: 7px;
  background: var(--color-neutral-200);
}
.om-profile > span {
  display: block;
  height: 100%;
  background: var(--color-accent-500);
}
.om-note {
  margin: var(--space-4) 0 0;
  font-size: 12px;
  max-width: 70ch;
  color: color-mix(in srgb, var(--color-text) 55%, transparent);
}
.om-disclosure {
  margin-top: var(--space-4);
  padding: var(--space-3) 0 0;
  border-top: 1px dashed var(--color-divider);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-accent-800);
}
.om-tools-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3) var(--space-6);
  align-items: end;
  margin-top: var(--space-4);
}
.om-search-field {
  flex: 1 1 28rem;
  min-width: min(100%, 20rem);
  max-width: 42rem;
  display: grid;
  gap: 4px;
}
.om-search-field label {
  font-size: 11px;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--color-neutral-600);
}
.bml-section-search, .bml-table-search, .bml-table-sort, .input {
  width: 100%;
  min-width: 0;
  min-height: 34px;
  padding: 6px 10px;
  font: inherit;
  font-size: 13px;
  color: var(--color-text);
  background: var(--color-surface);
  border: 1px solid var(--color-divider);
  border-radius: 0;
}
.bml-search-hint {
  margin: 0;
  font-size: 12px;
  line-height: 1.4;
  color: var(--color-neutral-600);
  max-width: none;
  white-space: normal;
}
.bml-search-status {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-neutral-600);
  min-height: 1.2em;
  flex: 0 0 auto;
  align-self: center;
}
.bml-table-wrap {
  margin-top: var(--space-2);
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
.bml-table-tools {
  display: flex; flex-wrap: wrap; gap: var(--space-2);
  margin: 0 0 var(--space-2);
}
.bml-table--fit {
  table-layout: fixed;
  width: 100%;
  max-width: 100%;
}
.bml-table--fit th,
.bml-table--fit td { min-width: 0; }
.bml-table--register .bml-col-sev,
.bml-table--register th.bml-col-sev { width: 56px; }
.bml-table--register .bml-col-key,
.bml-table--register th.bml-col-key { width: 18%; }
.bml-table--register .bml-col-detail,
.bml-table--register th.bml-col-detail { width: 38%; }
.bml-table--register .bml-col-evidence,
.bml-table--register th.bml-col-evidence { width: 28%; }
.bml-table--register .bml-col-note,
.bml-table--register th.bml-col-note { width: 8%; }
.bml-cell-wrap {
  min-width: 0;
  max-width: 100%;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.bml-cell-wrap--prose,
.bml-cell-wrap--evidence {
  white-space: normal;
  overflow: visible;
  text-overflow: unset;
}
.bml-col-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 0 0 4px;
  max-width: 100%;
}
.bml-col-chip {
  display: inline-block;
  max-width: 100%;
  padding: 1px 6px;
  border: 1px solid var(--color-divider);
  background: var(--color-neutral-100);
  font-size: 10.5px;
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: break-word;
  white-space: normal;
}
.bml-json {
  margin: var(--space-3) 0 0;
  padding: var(--space-3);
  overflow: auto;
  max-height: 420px;
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.45;
  background: var(--color-surface);
  border: 1px solid var(--color-divider);
  white-space: pre-wrap;
}
.bml-empty {
  margin: 0;
  padding: var(--space-6) 0;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-neutral-600);
  border-top: 1px dashed var(--color-divider);
  border-bottom: 1px dashed var(--color-divider);
}
.blueprint {
  position: relative;
  border: 1px solid var(--color-divider);
  border-radius: 0;
}
.blueprint > .corner {
  position: absolute; width: 11px; height: 11px;
  color: color-mix(in srgb, var(--color-text) 55%, transparent);
  pointer-events: none;
}
.blueprint > .corner::before, .blueprint > .corner::after {
  content: ""; position: absolute; background: currentColor;
}
.blueprint > .corner::before { left: 5px; top: 0; width: 1px; height: 100%; }
.blueprint > .corner::after  { top: 5px; left: 0; width: 100%; height: 1px; }
.blueprint > .corner.tl { top: -6px; left: -6px; }
.blueprint > .corner.tr { top: -6px; right: -6px; }
.blueprint > .corner.bl { bottom: -6px; left: -6px; }
.blueprint > .corner.br { bottom: -6px; right: -6px; }

.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  cursor: pointer; text-decoration: none;
  font-family: var(--font-heading); font-weight: var(--font-heading-weight);
  font-size: 14px; line-height: 1.2; color: var(--color-text);
  background: transparent; border: 1px solid var(--color-divider);
  padding: var(--space-2) calc(var(--space-3) * 1.2);
  border-radius: 0;
}
.btn:hover { background: color-mix(in srgb, var(--color-text) 7%, transparent); }
.btn-primary { background: var(--color-accent); color: var(--color-bg); border-color: var(--color-accent); }
.btn-primary:hover { background: var(--color-accent-600); color: var(--color-bg); text-decoration: none; }
.btn-secondary { border-color: var(--color-divider); }
.btn-ghost { color: var(--color-accent); border-color: transparent; padding-inline: var(--space-1); }
.btn.blueprint { position: relative; }

.tag {
  display: inline-flex; align-items: center; font-size: 11px;
  letter-spacing: .06em; padding: 3px 10px; border-radius: 0;
  text-transform: uppercase;
}
.tag-accent { background: var(--color-accent-100); color: var(--color-accent-800); }
.tag-neutral { background: var(--color-neutral-100); color: var(--color-neutral-800); }
.tag-outline { border: 1px solid var(--color-accent); color: var(--color-accent); }
.tag-blocking { background: var(--color-accent-800); color: var(--color-bg); }

.table { width: 100%; border-collapse: collapse; font-size: 14px; }
.table th {
  text-align: left; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: color-mix(in srgb, var(--color-text) 60%, transparent);
  padding: var(--space-2); border-bottom: 1px solid var(--color-divider);
}
.table td {
  padding: var(--space-2);
  border-bottom: 1px solid color-mix(in srgb, var(--color-text) 8%, transparent);
  vertical-align: top;
}
.table tbody tr:hover { background: color-mix(in srgb, var(--color-text) 4%, transparent); }
.table.bml-data-table th { cursor: default; }

.om-footer {
  margin-top: var(--space-8);
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-divider);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-neutral-600);
}

@media (max-width: 860px) {
  .om-shell { padding: var(--space-4); }
  .om-header { grid-template-columns: 1fr; }
  .om-header__title { padding-right: 0; }
  .om-tools {
    justify-content: flex-start;
    margin-left: 0;
    padding-left: 0;
    border-left: 0;
    padding-top: var(--space-3);
    border-top: 1px solid var(--color-divider);
  }
  .om-kpi { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .om-spine { grid-template-columns: 28px minmax(0, 1fr); gap: var(--space-3); }
  .om-search-field { flex-basis: 100%; max-width: none; }
  .om-methods__grid, .om-assumptions, .om-ledger { grid-template-columns: 1fr; }
  .om-bar-row {
    grid-template-columns: minmax(3.5rem, 24%) minmax(2.5rem, 1fr) minmax(6.75rem, max-content);
    min-width: 18rem;
  }
  h1 { font-size: 32px; }
}

@media print {
  a[href="#"], button, .om-tools-row, .bml-table-tools, .bml-skip-link,
  .om-panel__tools, .om-jump { display: none !important; }
  html, body { background: #fff; }
  .om-shell { max-width: none; padding: 0; }
  article, section, .om-spine, .om-method, .om-assumption-card { break-inside: avoid; }
  .om-group__summary { position: static; }
}
"""
)

COCKPIT_JS = """
(() => {
  const sectionInput = document.getElementById("bml-section-search");
  const sectionStatus = document.getElementById("bml-section-search-status");
  const spines = Array.from(document.querySelectorAll("[data-spine-section]"));

  function filterSections() {
    if (!sectionInput) return;
    const query = sectionInput.value.trim().toLowerCase();
    let visible = 0;
    for (const node of spines) {
      const labeled = node.getAttribute("data-search") || "";
      const hay = `${labeled} ${node.textContent || ""}`.toLowerCase();
      const show = !query || hay.includes(query);
      node.hidden = !show;
      if (show) visible += 1;
    }
    if (sectionStatus) {
      sectionStatus.textContent = query
        ? `${visible} of ${spines.length} sections match`
        : "";
    }
  }
  if (sectionInput) {
    sectionInput.addEventListener("input", filterSections);
    sectionInput.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        sectionInput.value = "";
        filterSections();
      }
    });
  }

  function bindPanelFilter(panel) {
    const kind = panel.getAttribute("data-panel");
    const input = panel.querySelector(`[data-filter-target="${kind}"]`);
    const count = panel.querySelector(`[data-filter-count="${kind}"]`);
    if (!input) return;

    const apply = () => {
      const query = input.value.trim().toLowerCase();
      if (kind === "assumptions") {
        const cards = Array.from(panel.querySelectorAll("[data-assumption-card]"));
        let shown = 0;
        for (const card of cards) {
          const hay = `${card.getAttribute("data-search") || ""} ${card.textContent || ""}`.toLowerCase();
          const match = !query || hay.includes(query);
          card.hidden = !match;
          if (match) shown += 1;
        }
        for (const group of panel.querySelectorAll("[data-assumption-group]")) {
          const any = Array.from(group.querySelectorAll("[data-assumption-card]")).some((card) => !card.hidden);
          group.hidden = Boolean(query) && !any;
          if (query && any) group.open = true;
        }
        if (count) {
          count.textContent = query
            ? `${shown} of ${cards.length} notes match`
            : `${cards.length} notes`;
        }
        return;
      }
      if (kind === "ledger") {
        const rows = Array.from(panel.querySelectorAll("[data-ledger-row]"));
        let shown = 0;
        for (const row of rows) {
          const hay = `${row.getAttribute("data-search") || ""} ${row.textContent || ""}`.toLowerCase();
          const match = !query || hay.includes(query);
          row.hidden = !match;
          if (match) shown += 1;
        }
        for (const group of panel.querySelectorAll("[data-ledger-group]")) {
          const any = Array.from(group.querySelectorAll("[data-ledger-row]")).some((row) => !row.hidden);
          group.hidden = Boolean(query) && !any;
          if (query && any) group.open = true;
        }
        if (count) {
          count.textContent = query
            ? `${shown} of ${rows.length} metrics match`
            : `${rows.length} metrics`;
        }
      }
    };

    input.addEventListener("input", apply);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        input.value = "";
        apply();
      }
    });
  }
  document.querySelectorAll("[data-panel]").forEach(bindPanelFilter);

  function bindTable(wrap) {
    const table = wrap.querySelector("table.bml-data-table");
    if (!table) return;
    const body = table.tBodies[0];
    if (!body) return;
    const rows = Array.from(body.rows);
    const search = wrap.querySelector(".bml-table-search");
    const sortSelect = wrap.querySelector(".bml-table-sort");
    const apply = () => {
      const q = (search && search.value || "").trim().toLowerCase();
      rows.sort((a, b) => {
        if (!(sortSelect && sortSelect.value)) return 0;
        const idx = Number(sortSelect.value);
        const av = (a.cells[idx] && a.cells[idx].textContent || "").trim();
        const bv = (b.cells[idx] && b.cells[idx].textContent || "").trim();
        const an = Number(av.replace(/,/g, ""));
        const bn = Number(bv.replace(/,/g, ""));
        if (!Number.isNaN(an) && !Number.isNaN(bn) && av !== "" && bv !== "") {
          return an - bn;
        }
        return av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
      });
      for (const row of rows) {
        const text = row.textContent.toLowerCase();
        row.hidden = Boolean(q) && !text.includes(q);
        body.appendChild(row);
      }
    };
    if (search) {
      search.addEventListener("input", apply);
      search.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          search.value = "";
          apply();
        }
      });
    }
    if (sortSelect) sortSelect.addEventListener("change", apply);
  }
  document.querySelectorAll(".bml-table-wrap").forEach(bindTable);

  const printBtn = document.getElementById("bml-print");
  if (printBtn) printBtn.addEventListener("click", () => window.print());

  const csvBtn = document.getElementById("bml-csv");
  const csvPayload = document.getElementById("bml-csv-payload");
  if (csvBtn && csvPayload) {
    csvBtn.addEventListener("click", () => {
      const blob = new Blob([csvPayload.textContent || ""], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = csvBtn.getAttribute("data-filename") || "eda-findings.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });
  }
})();
"""
