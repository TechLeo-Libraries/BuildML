/** Minimal Lucide-style path set for navigation and affordances. */

const PATHS = {
  gauge: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm0 0v3m8.5-5.5a8.5 8.5 0 1 0-17 0",
  "shield-check": "M12 3 4 6v6c0 5 3.5 8.5 8 10 4.5-1.5 8-5 8-10V6l-8-3Zm-1 12-3-3 1.4-1.4L11 12.2l3.6-3.6L16 10l-5 5Z",
  "bar-chart-3": "M3 3v18h18M8 17V9m5 8V5m5 12v-6",
  "git-branch": "M6 3v12m0 0a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm12-6a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm0 0v2a4 4 0 0 1-4 4H9",
  layers: "m12 2 9 4.5-9 4.5L3 6.5 12 2Zm0 9 9 4.5-9 4.5-9-4.5L12 11Z",
  crosshair: "M12 2v4m0 12v4M2 12h4m12 0h4m-4 0a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z",
  "scan-search": "M3 7V5a2 2 0 0 1 2-2h2m10 0h2a2 2 0 0 1 2 2v2M3 17v2a2 2 0 0 0 2 2h2m10 0h2a2 2 0 0 0 2-2v-2m-5-5a4 4 0 1 1-8 0 4 4 0 0 1 8 0Zm3 3 2.5 2.5",
  image: "M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm0 11 4-4 3 3 5-6 4 7",
  "graduation-cap": "M22 10 12 5 2 10l10 5 10-5Zm-6.5 3.2v4.3c0 1.2-2 2.5-3.5 2.5S8.5 18.7 8.5 17.5v-4.3",
  search: "m21 21-4.3-4.3M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z",
  "file-down": "M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-6Zm0 0v6h6M12 18v-6m0 6-2.5-2.5M12 18l2.5-2.5",
  moon: "M21 14.5A8.5 8.5 0 1 1 9.5 3 7 7 0 0 0 21 14.5Z",
  sun: "M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7 1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2-1.4-1.4M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
  menu: "M4 7h16M4 12h16M4 17h16",
  x: "M6 6l12 12M18 6 6 18",
  expand: "M9 3H3v6m12-6h6v6M9 21H3v-6m18 0v6h-6",
  download: "M12 3v12m0 0-4-4m4 4 4-4M5 21h14",
  alert: "M12 9v4m0 4h.01M10.3 4.3 2.6 18a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0Z",
  book: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2V5Zm0 14a2 2 0 0 1 2-2h13",
  link: "M10 13a5 5 0 0 0 7.07 0l1.41-1.41a5 5 0 0 0-7.07-7.07L10 5.93m4 4.14a5 5 0 0 0-7.07 0L5.5 11.48a5 5 0 1 0 7.07 7.07L14 17.14",
  "list-checks": "M3 6h.01M3 12h.01M3 18h.01M8 6h13M8 12h13M8 18h8M3.5 6l.8.8L6 5",
};

export function iconSvg(name, className = "icon") {
  const d = PATHS[name] || PATHS.alert;
  return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;
}

export function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => {
    const name = node.getAttribute("data-icon");
    node.innerHTML = iconSvg(name);
  });
}
