# Folio Claims NLP

**Tier B** cross-domain product proof: NLP claim-note routing + CBR case
memory + symbolic escalation guardrails for synthetic P&C claims.

## Product narrative

Folio routes free-text claim notes to desks, retrieves similar past cases for
escalate-or-not, and induces explainable policy rules:

1. Profiles the note corpus and fits TF-IDF + logistic (+ NMF topics)
2. Builds CBR case memory from structured claim features (train only)
3. Induces symbolic decision-tree guardrails on the same case split

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\folio-claims-nlp\script.py
```

## Leakage controls

- NLP stratified split before TF-IDF / topics fit
- CBR case memory built from train cases only
- Symbolic rules induced on the same train split as CBR
- Test text / CBR / symbolic eval after each stage locks

## What fails if leakage is ignored

- Fitting the text vectorizer on test notes invents desk accuracy
- Putting test claims into CBR memory makes escalate accuracy meaningless
- Inducing guardrail rules on the full book looks more “compliant” than production

## Upstream Tier A building blocks

`ticket-routing-nlp`, `case-memory-claims`, `warranty-cbr-memory`,
`policy-rules-neuro-symbolic`, `claim-severity-regression`

## Limitations

Claim notes reuse synthetic ticket language: not a real P&C extract.
