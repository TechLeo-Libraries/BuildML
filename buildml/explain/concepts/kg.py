# ruff: noqa: E501
"""Knowledge-graph concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

KG_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="kg-triples",
            title="Knowledge graphs as (head, relation, tail) triples",
            summary=(
                "fit_kg learns from train rows of (head, relation, tail); "
                "vocabularies and adjacency are train-only."
            ),
            definition=(
                "A knowledge graph here is a set of typed edges stored as "
                "triples (h, r, t). Session rows are triples; splits partition "
                "triples. Embeddings and symbolic queries use the train store."
            ),
            intuition=(
                "Alice —works_at→ Acme is one triple. Link prediction asks "
                "which tails complete (Alice, works_at, ?); symbolic query "
                "walks the train edges without an LLM."
            ),
            formal_idea=("KG = {(h, r, t)}; embeddings e_h, e_r, e_t; score s(h,r,t)."),
            why_it_matters=(
                "Train-only vocab prevents holdout entities from shaping embeddings.",
                "Distinct from adjacency+features node classification.",
            ),
            how_buildml_uses=(
                "Session.fit_kg(method='transe'|'distmult', "
                "head_column=..., relation_column=..., tail_column=...).",
            ),
            interpretation_rules=(
                "Prefer evaluate_kg filtered MRR / Hits@K.",
                "query_kg never sees holdout triples.",
            ),
            assumptions=(
                "Explicit head/relation/tail columns; split present; ≥2 entities.",
            ),
            failure_modes=(
                "Fitting on full-frame triples; tiny graphs; OOV-heavy holdout.",
            ),
            anti_patterns=(
                "Calling this Neo4j / a graph-DB product.",
                "Confusing with set_graph / fit_graph node classification.",
                "Confusing with RAG retrieve/generate.",
            ),
            worked_example_pattern=(
                "split → fit_kg(method='transe', head_column=..., "
                "relation_column=..., tail_column=...) → evaluate_kg(k=10).",
            ),
            related_concepts=(
                "kg-transe-distmult",
                "kg-link-prediction",
                "kg-symbolic-query",
                "kg-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="kg-transe-distmult",
            title="KG embedding backends (native + PyKEEN)",
            summary=(
                "backend='native': numpy TransE/DistMult. backend='pykeen': "
                "RotatE/ComplEx/TransE/DistMult via buildml[kg-industry]. "
                "Both use margin ranking with disclosed negative sampling."
            ),
            definition=(
                "Native TransE scores −‖h+r−t‖; DistMult scores ⟨h,r,t⟩. "
                "PyKEEN adds RotatE (complex rotation) and ComplEx "
                "(complex trilinear). Negatives corrupt head/tail on train triples."
            ),
            intuition=(
                "TransE treats a relation as a vector you add to the head to "
                "reach the tail. RotatE rotates head embeddings in complex space. "
                "ComplEx uses complex-valued bilinear products."
            ),
            formal_idea=(
                "min Σ max(0, γ − s(pos) + s(neg)) with disclosed neg_ratio; "
                "PyKEEN uses sLCWA/LCWA on train factory."
            ),
            why_it_matters=(
                "Industry-standard KGE models when PyKEEN installed; "
                "numpy fallback stays core-only.",
                "Negative sampling must be disclosed; holdout never corrupted in.",
            ),
            how_buildml_uses=(
                "Session.fit_kg(backend='native'|'pykeen', method='transe'|"
                "'distmult'|'rotate'|'complex', embedding_dim=..., "
                "epochs=..., neg_ratio=...).",
            ),
            interpretation_rules=(
                "Read disclosures for backend, neg_ratio, and scoring formula.",
                "Loss alone is not ranking quality — use evaluate_kg.",
            ),
            assumptions=("Dense embeddings fit in memory for the train catalog.",),
            failure_modes=("Too few epochs; collapsed embeddings; tiny entity sets.",),
            anti_patterns=(
                "Requiring Neo4j or PyG for this Session path.",
                "Training negatives from test triples.",
            ),
            worked_example_pattern=(
                "fit_kg(backend='pykeen', method='rotate', epochs=50) → "
                "predict_links(mode='tail')."
            ,),
            related_concepts=("kg-triples", "kg-link-prediction"),
        ),
        _note(
            key="kg-link-prediction",
            title="Link prediction (score / predict / evaluate)",
            summary=(
                "score_triples scores full triples; predict_links fills "
                "tail|head|relation; evaluate_kg reports filtered MRR and Hits@K."
            ),
            definition=(
                "Filtered ranking removes other known true triples from the "
                "candidate list before computing the rank of the true fill-in."
            ),
            intuition=(
                "Ask which entities complete a partial fact, then check whether "
                "the true answer ranks near the top among train catalog entities."
            ),
            formal_idea=(
                "For each holdout (h,r,t): rank t among entities for (h,r,?) "
                "and h among entities for (?,r,t); average MRR / Hits@K."
            ),
            why_it_matters=(
                "Standard KG protocol; comparable across TransE/DistMult.",
                "Relation prediction is available via mode='relation'.",
            ),
            how_buildml_uses=(
                "predict_links(mode='tail'|'head'|'relation'); "
                "evaluate_kg(partition='test', k=...).",
            ),
            interpretation_rules=(
                "OOV holdout entities/relations are skipped and disclosed.",
                "Do not equate with Graph ML node accuracy or RAG nDCG.",
            ),
            assumptions=("Holdout triples mostly in train vocab (known-entity protocol).",),
            failure_modes=("All-OOV holdout; k larger than catalog.",),
            anti_patterns=("Reporting raw loss as link-prediction quality.",),
            worked_example_pattern=(
                "evaluate_kg(k=10) → inspect metrics['mrr'] / hits_at_10."
            ,),
            related_concepts=("kg-triples", "kg-transe-distmult", "leakage-boundary"),
        ),
        _note(
            key="kg-symbolic-query",
            title="Symbolic neighborhood / path / typed query",
            summary=(
                "query_kg walks the train adjacency: neighbors, typed "
                "(entity, relation, ?), or shortest path — not an LLM."
            ),
            definition=(
                "Symbolic queries operate on the materialised train triple "
                "store (out/in adjacency). Paths use BFS capped by max_hops."
            ),
            intuition=(
                "Ask 'who does Alice work with?' or 'is there a path from A to B?' "
                "using stored edges — not generated text."
            ),
            formal_idea=(
                "neighbors(v); typed(v,r); path(s,t) via BFS on directed out-edges."
            ),
            why_it_matters=(
                "Complements embedding link prediction with exact structure queries.",
                "Holdout edges are intentionally invisible.",
            ),
            how_buildml_uses=(
                "query_kg(mode='neighbors'|'typed'|'path', entity=..., "
                "relation=..., source=..., target=...).",
            ),
            interpretation_rules=(
                "Empty path means no train path within max_hops — not model failure.",
            ),
            assumptions=("KgPlan with train adjacency from fit_kg.",),
            failure_modes=("OOV entities; disconnected train graphs.",),
            anti_patterns=(
                "Treating query_kg as Cypher/Neo4j or as RAG retrieve.",
            ),
            worked_example_pattern=(
                "query_kg(mode='typed', entity='Alice', relation='works_at')."
            ,),
            related_concepts=("kg-triples", "kg-link-prediction"),
        ),
        _note(
            key="kg-bundle-boundary",
            title="KG bundle vs Session checkpoint / Graph ML / RAG",
            summary=(
                "save_kg_bundle stores KgPlan as buildml.kg_bundle.v1; "
                "checkpoints, Graph ML bundles, and RAG bundles are separate."
            ),
            definition=(
                "A KG bundle persists train vocabularies, embeddings, and "
                "adjacency. It does not embed the dataset or replace a Session "
                "checkpoint, GraphPlan, or RAG index."
            ),
            intuition=(
                "Reload workflow state with checkpoint_load; reload the KG "
                "with load_kg_bundle."
            ),
            formal_idea=("buildml.kg_bundle.v1 = meta.json + kg_plan.joblib."),
            why_it_matters=("Prevents silent mixing of artifact types.",),
            how_buildml_uses=("Session.save_kg_bundle / load_kg_bundle.",),
            interpretation_rules=(
                "Bundles are complementary to checkpoints, not interchangeable.",
            ),
            assumptions=("A KgPlan exists.",),
            failure_modes=("Expecting Neo4j dump semantics inside the bundle.",),
            anti_patterns=(
                "Loading a Graph ML or RAG bundle via load_kg_bundle.",
            ),
            worked_example_pattern=(
                "save_kg_bundle(path) → load_kg_bundle(path) → evaluate_kg()."
            ,),
            related_concepts=("kg-triples", "checkpoint-boundary"),
        ),
    )
}
