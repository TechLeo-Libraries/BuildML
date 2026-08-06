// Curriculum barrel. The core stage 1–4 entries, the framing/interpretation
// stages that bracket them, the stage 1–4 additions, and the depth layer that
// extends each core entry's prose and adds its reading procedure and decision.

import { CONCEPTS as CORE, STAGES as CORE_STAGES } from './eda-concepts-core.js';
import { CONCEPTS_ENDS } from './eda-concepts-ends.js';
import { CONCEPTS_ADD_A } from './eda-concepts-add-a.js';
import { CONCEPTS_ADD_B } from './eda-concepts-add-b.js';
import { DEPTH_A } from './eda-depth-a.js';
import { DEPTH_B } from './eda-depth-b.js';

const DEPTH = Object.assign({}, DEPTH_A, DEPTH_B);

const deepened = {};
Object.keys(CORE).forEach(slug => {
  const base = CORE[slug];
  const d = DEPTH[slug];
  deepened[slug] = d
    ? Object.assign({}, base, {
        prose: base.prose.concat(d.more || []),
        read: d.read,
        decide: d.decide,
      })
    : base;
});

// Stage order is the reading order: framing, then the numeric stages, then
// interpretation. Within a stage, core entries precede the later additions.
const byStage = src => stage => Object.keys(src).filter(k => src[k].stage === stage);

const ORDERED = {};
[0, 1, 2, 3, 4, 5].forEach(stage => {
  [CONCEPTS_ENDS, deepened, CONCEPTS_ADD_A, CONCEPTS_ADD_B].forEach(src => {
    byStage(src)(stage).forEach(slug => { ORDERED[slug] = src === deepened ? deepened[slug] : src[slug]; });
  });
});

export const CONCEPTS = ORDERED;

export const STAGES = [
  { n: '00', key: 0, label: 'Framing', blurb: 'what the data would have to be before any statistic means anything' },
  ...CORE_STAGES,
  { n: '05', key: 5, label: 'Interpretation', blurb: 'what a fitted model may be said to show, and what ships with it' },
];
