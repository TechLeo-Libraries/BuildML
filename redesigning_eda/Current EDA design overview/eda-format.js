// Shared formatting helpers for the EDA engine and concept library.

export const fmt = {
  n: v => (v == null ? '—' : Number(v).toLocaleString('en-US')),
  pct: (v, d = 3) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`),
  dec: (v, d = 5) => (v == null ? '—' : Number(v).toFixed(d)),
  short: (v, d = 4) => (v == null ? '—' : Number(v).toFixed(d).replace(/^0/, '')),
  compact: v => {
    if (v == null) return '—';
    const a = Math.abs(v);
    if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (a >= 1e4) return `${Math.round(v / 1e3)}k`;
    if (a >= 10) return String(Math.round(v));
    if (a >= 1) return v.toFixed(1);
    if (a === 0) return '0';
    return v.toFixed(3).replace(/^0/, '');
  },
};

export const plural = (n, w, s) => `${w}${n === 1 ? '' : (s || 's')}`;

export const list = (xs, max = 4) => {
  if (!xs || !xs.length) return '';
  const shown = xs.slice(0, max).join(', ');
  return xs.length > max ? `${shown} and ${xs.length - max} more` : shown;
};

export const truncate = (s, n = 15) => (String(s).length > n ? `${String(s).slice(0, n - 1)}…` : String(s));

export const names = (xs, max = 4) => list((xs || []).map(x => (typeof x === 'string' ? x : x.name)), max);
