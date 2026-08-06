// Two sample dataset descriptors. Raw observed statistics only —
// every sentence, finding, chart and recommendation is derived in eda-engine.js.

export const churn = {
  label: 'churn-2026-08',
  session: 'session/churn-2026-08',
  engine: 'pandas',
  version: '2.4.0',
  runtimeNote: 'has_native = false · lazy = false',
  rows: 800,
  rowsTotal: 800,
  completeRows: 617,
  vifThreshold: 5,
  columns: [
    { name: 'customer_id', role: 'id', dtype: 'integer', distinct: 800, missingRate: 0 },
    { name: 'age', role: 'feature', dtype: 'integer', distinct: 58, missingRate: 0.07875, mi: 0.0173094, vif: 1.88203 },
    { name: 'income', role: 'feature', dtype: 'float', distinct: 713, missingRate: 0.10875, mi: 0.0224417, vif: 5.21204 },
    { name: 'monthly_spend', role: 'feature', dtype: 'float', distinct: 781, missingRate: 0.05125, mi: 0.0388102, vif: 6.41822 },
    { name: 'tenure_years', role: 'feature', dtype: 'float', distinct: 204, missingRate: 0, mi: 0.0101932, vif: 1.34117 },
    { name: 'risk_score', role: 'feature', dtype: 'float', distinct: 742, missingRate: 0, mi: 0.0421364, vif: 3.10411, drift: true },
    { name: 'plan_tier', role: 'feature', dtype: 'categorical', distinct: 3, missingRate: 0, mi: 0.0087412, drift: true },
    { name: 'signup_channel', role: 'feature', dtype: 'categorical', distinct: 4, missingRate: 0.25125, mi: 0.0029103 },
    { name: 'city', role: 'feature', dtype: 'categorical', distinct: 12, missingRate: 0.14, mi: 0.0041213, drift: true },
    { name: 'constant_flag', role: 'feature', dtype: 'boolean', distinct: 1, missingRate: 0, mi: 0 },
    { name: 'target_churn', role: 'target', dtype: 'boolean', distinct: 2, missingRate: 0 },
  ],
  target: {
    name: 'target_churn', task: 'binary',
    classes: [{ label: 'false', count: 613 }, { label: 'true', count: 187 }],
  },
  driftDetail: {
    column: 'plan_tier', testRows: 160,
    levels: [
      { label: 'basic', train: 0.49, test: 0.38 },
      { label: 'plus', train: 0.35, test: 0.41 },
      { label: 'pro', train: 0.16, test: 0.21 },
    ],
  },
  anomalies: {
    scored: 617, flagged: 31, contamination: 0.05, cutIndex: 8,
    histogram: [8, 42, 96, 142, 131, 89, 52, 26, 17, 14],
  },
};

/* ── a wide, messy, regression frame ──────────────────────────────── */

function mulberry(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildWide() {
  const rnd = mulberry(20260805);
  const rows = 5200;
  const cols = [];

  cols.push({ name: 'flight_uid', role: 'id', dtype: 'string', distinct: rows, missingRate: 0 });
  cols.push({ name: 'booking_reference', role: 'feature', dtype: 'string', distinct: rows, missingRate: 0, mi: 0.000412 });

  const numeric = [
    'sched_block_minutes', 'taxi_out_minutes', 'taxi_in_minutes', 'distance_km', 'aircraft_age_years',
    'seats_sold', 'seats_available', 'load_factor', 'crew_count', 'fuel_burn_kg',
    'payload_kg', 'headwind_kts', 'visibility_m', 'ceiling_ft', 'precip_mm',
    'gate_turnaround_minutes', 'prior_leg_delay_minutes', 'maint_events_30d', 'atc_hold_minutes', 'runway_queue_len',
  ];
  const categorical = [
    'carrier_code', 'origin_iata', 'dest_iata', 'aircraft_type', 'haul_band',
    'departure_slot', 'weekday', 'season', 'wx_category', 'gate_area',
    'crew_base', 'ops_region', 'fare_class_mix', 'booking_channel',
  ];
  const flags = ['is_redeye', 'is_codeshare', 'legacy_ff_flag', 'pilot_bid_locked'];

  const missingPlan = {
    precip_mm: 0.412, ceiling_ft: 0.331, visibility_m: 0.298, prior_leg_delay_minutes: 0.244,
    atc_hold_minutes: 0.187, gate_turnaround_minutes: 0.142, headwind_kts: 0.118,
    fare_class_mix: 0.104, crew_base: 0.087, maint_events_30d: 0.063,
    gate_area: 0.051, payload_kg: 0.038, aircraft_age_years: 0.024,
    runway_queue_len: 0.019, booking_channel: 0.012, fuel_burn_kg: 0.006,
  };

  const strongVif = { sched_block_minutes: 18.4122, distance_km: 16.9037, fuel_burn_kg: 11.2418, seats_available: 8.7741, seats_sold: 7.9903, load_factor: 6.2216, payload_kg: 5.4408 };

  numeric.forEach((name, i) => {
    const mi = 0.0004 + rnd() * 0.075;
    cols.push({
      name, role: 'feature', dtype: i % 4 === 0 ? 'integer' : 'float',
      distinct: 120 + Math.floor(rnd() * 3400),
      missingRate: missingPlan[name] || 0,
      mi: name === 'prior_leg_delay_minutes' ? 0.184206 : (name === 'atc_hold_minutes' ? 0.121337 : mi),
      vif: strongVif[name] != null ? strongVif[name] : 1.05 + rnd() * 3.4,
      drift: name === 'prior_leg_delay_minutes' || name === 'gate_turnaround_minutes',
    });
  });

  categorical.forEach((name, i) => {
    cols.push({
      name, role: 'feature', dtype: 'categorical',
      distinct: name === 'origin_iata' || name === 'dest_iata' ? 148 + i : 3 + Math.floor(rnd() * 11),
      missingRate: missingPlan[name] || 0,
      mi: 0.0006 + rnd() * 0.042,
      drift: name === 'departure_slot',
    });
  });

  flags.forEach((name, i) => {
    const constant = i >= 2;
    cols.push({
      name, role: 'feature', dtype: 'boolean',
      distinct: constant ? 1 : 2, missingRate: 0,
      mi: constant ? 0 : 0.002 + rnd() * 0.01,
    });
  });

  cols.push({ name: 'arr_delay_minutes', role: 'target', dtype: 'float', distinct: 431, missingRate: 0 });

  const missingCells = Math.round(cols.reduce((s, c) => s + (c.missingRate || 0) * rows, 0));

  return {
    label: 'ops-delay-2026-07',
    session: 'session/ops-delay-2026-07',
    engine: 'polars',
    version: '2.4.0',
    runtimeNote: 'has_native = true · lazy = true',
    rows,
    rowsTotal: 41800,
    completeRows: 1284,
    vifThreshold: 5,
    columns: cols,
    target: {
      name: 'arr_delay_minutes', task: 'regression',
      stats: { min: -22, median: 7, max: 486, skew: 3.41 },
      histogram: [214, 1180, 1642, 892, 517, 328, 194, 121, 68, 24, 12, 8],
    },
    driftDetail: {
      column: 'departure_slot', testRows: 1040,
      levels: [
        { label: 'early_am', train: 0.22, test: 0.14 },
        { label: 'am_peak', train: 0.26, test: 0.21 },
        { label: 'midday', train: 0.18, test: 0.19 },
        { label: 'pm_peak', train: 0.21, test: 0.29 },
        { label: 'evening', train: 0.10, test: 0.13 },
        { label: 'redeye', train: 0.03, test: 0.04 },
      ],
    },
    anomalies: {
      scored: 1284, flagged: 64, contamination: 0.05, cutIndex: 9,
      histogram: [21, 96, 214, 318, 271, 164, 88, 42, 26, 27, 12, 5],
    },
    _missingCells: missingCells,
  };
}

export const opsDelay = buildWide();
