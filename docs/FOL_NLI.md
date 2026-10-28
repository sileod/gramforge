# Reproducible FOL-NLI with Gramforge

The maintained generator and postprocessor live in `src/gramforge/fol_nli/`.
They use the current `gramforge.grammars.FOL.FOL_grammar` factory with an explicit
FOL-NLI profile. The goal is reproducible generation with corrected semantics,
explicit modeling assumptions and auditable labels. Better downstream transfer
has not yet been established.

## Scope and semantics

The profile retains universal and existential quantification; negative and
restricted quantifiers; Boolean combinations, equivalence, XOR, unless and
if/otherwise; counting through five; named people; active/passive relations;
propositional statements; and contextual property reuse. It restores original
comparison, superlative, list and predicate-reuse constructions absent or
unreachable in the current base grammar. Generated blocks have 1–32 premise
units. A unit may contain more than one English line; it is never identified by
line number.

The domain is nonempty, consists of people, and is not enumerated or bounded.
Only an explicitly closed room is limited to its stated members. Additional
unnamed people remain possible. Distinctness applies to names actually mentioned,
not to seven silent background constants. The visible setup states the person
and distinct-name conventions. `background` is included in every exported row.

Comparisons use subject-first arguments, strict transitive order, opposite
comparative inverses and monotonic adjective thresholds. Superlatives include
group membership and strict superiority over other members. Sibling is symmetric
and irreflexive. When like/hate relations occur, their mutual exclusion is
introduced as an English/formula definition premise with its own stable ID:
“Definition: For any people A and B, possibly the same person, A cannot both
like and hate B. Neither relation is required, and for different people these
relations need not be mutual.” It is
absent from the hidden background. It remains in every shortened example from
the same premise group, whether or not a proof uses it. This avoids a cue from
its presence or absence. This is an
explicit task assumption, not a universal claim about human psychology.

Dream statements remain excluded: material implication is not an adequate
semantics for dream content. Ambiguous combinations such as negative existential
“no one ... who A ... B” are rejected rather than treated as the negation of
`A -> B`. Quantification and conditional reasoning remain available separately
and in their unambiguous compositions. Every such departure is deliberate;
coverage is not inferred merely from the number of grammar productions.

The base grammar now establishes person typing for named room occupants and
makes `outside_entity` productive again. Remaining FOL-NLI changes are explicit
in the profile factory. English names, 145 predicate phrases and ten proposition
phrases are frozen in `vocabulary.json`, recovered from the historical `unigram_resources.py` resource.
They are sampled deterministically per group. Formula symbols and logical
negations are never replaced by English strings.

## Build and audit

Use Python 3.12, Vampire 4.8, and the pinned additional dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements-fol-nli.txt
python -m pytest -q tests/test_fol_nli_semantics.py tests/test_fol_grammar_quality.py
python -m gramforge.fol_nli.build --output outputs/fol-nli/pool --groups 100000 --workers 32 --hypotheses 12 --seconds 2 --vampire /path/to/vampire
python -m gramforge.fol_nli.inspect outputs/fol-nli/pool
python -m gramforge.fol_nli.curate --input outputs/fol-nli/pool --output outputs/fol-nli/corrected --size 100000
python -m gramforge.fol_nli.verify outputs/fol-nli/corrected --pool outputs/fol-nli/pool --workers 24 --independent-sample 1000 --vampire /path/to/vampire
```

Cache the tokenizer at its pinned revision before curation. The default is
`sileod/deberta-v3-base-tasksource-nli`, revision
`3209a6ab012eab725e8f24547972f9aa133d1345`; it is used only to count tokens.
No model inference or LLM labeling is used to construct labels.

The pool size is a candidate budget, not a promise of 100k usable examples.
If proof-support bins or split/label cells are too small, curation fails with
counts. Extend the pool by rerunning the same build with a larger `--groups` budget;
do not silently relax the requested balancing. Generation failures, inconsistent premises and unresolved solver
results are retained as diagnostics and never exported as labels.

A small end-to-end plumbing check can explicitly request `--size 30
--heuristic-size 6 --reference-bin 1`; that is a smoke test, not the paper's
five-input balancing protocol or a training-quality result.

The reference bin is a selection policy, not a change to logical expressivity.
An explicitly named available-depth comparison may instead use
`--reference-bin 1 --heuristic-size 1000 --size 50000`. It retains the same full
grammar, logical checks, deduplication, difficulty filtering and surface matching,
but its proof-depth distribution differs from five-input-bin balancing. Report
that difference and use the same training count in its published-data control.
Neither policy silently substitutes for the other if capacity is insufficient.

## Notebook postprocessing recovered

The provenance reference is `unigram v4 postprocess unigram-fol.ipynb`, especially
cells 2, 5, 11, 18, 22, 25, 60, 82–86. The notebook is historical provenance;
it is not required at runtime. All runtime vocabulary is included here.

The pipeline implements:

1. Premise satisfiability filtering and both hypothesis polarities. An unknown
   result is never interpreted as neutral.
2. Same-concept hypothesis matching and context-derived questions, including
   named instantiations of generated properties.
3. Native Vampire proof-input and inference-trace extraction. SAT-portfolio
   proofs with valid premise inputs and an inference trace are kept directly.
   Only missing support triggers a supplementary proof-search run, with a
   budget of at least ten seconds (five times the classification budget).
   A failed supplement never erases the portfolio proof or changes its label.
   Group/progress counters report proof sources, supplement attempts and
   missing support, counting actual searches rather than cached opposite rows.
4. Syntactic canonical deduplication with bound-variable normalization and
   commutative operand ordering and double-negation elimination. Questions
   directly stated by a whole premise, or its explicit opposite, are excluded
   after both query polarities have been generated. This is not general
   semantic-equivalence deduplication.
5. A global 5,000-example proof-trace cap, seeded traversal before applying the
   cap, and explicit exclusion of premise-independent nonneutral questions.
   Each E/C query must have a Z3 countermodel under the background **without
   any premise**, including definition premises. UNSAT means independent and
   unknown means uncertified; both are excluded, with separate counters. A proof
   that mentions premises is not sufficient because proofs can be nonminimal.
   Opposite pairs share the dependence check. Final verification repeats it.
6. Proof-preserving premise shortening and distractor shuffling on paired
   clause records. Setup and definitions are mandatory across all questions in
   a group, and distractors fill the same group-level clause budget for every
   label. A proof exceeding that budget is excluded, not given a longer input.
   Stable IDs keep multi-line conditionals intact.
7. Full-input token-length filtering at 500 tokens; no token truncation.
8. A 100-tree gradient-boosting difficulty heuristic trained on 1,000 stratified
   reserved examples. Their complete premise groups are excluded from final splits.
9. Used-premise balancing capped by the five-input reference bin, final label
   balancing and 80/10/10 splits grouped by the original full premise. Within
   each split, labels share counts in joint public-text buckets: definition,
   room and occupant presence/absence, generic negation wrapper, premise length
   above/below 120 words and hypothesis length above/below 12 words. Integer
   rounding differs by at most one row per bucket. Missing matched capacity
   fails curation rather than relaxing quotas; final verification checks counts.
10. Fresh verification of the final exported text/formula pairs and labels,
    with independent open-domain Z3 checks of a fixed sample.

The historical notebook contains several overwritten/experimental stages; this
pipeline fixes one explicit protocol instead of replaying execution history.
Proof-support counts are Vampire input counts, not claims of subset-minimality.

## Reproducibility and speed

Each group is seeded by experiment seed and group index. No rendering uses an
object ID or reseeds NumPy from system entropy. Per-group compressed checkpoints
make generation resumable. The specification pins source and solver hashes and
dependency versions; changed sources require a fresh output directory. Each pool
archives its exact Python sources and vocabulary in `source.zip`.
The export separately archives its complete curation sources and hashes, so a
new curation revision over a frozen candidate pool remains reproducible.

Grammar factories and generation-height bounds are reused per worker. Explicit
backtracking checkpoints avoid rejecting most long blocks after one failed
choice. Opposite hypotheses reuse exactly swapped satisfiability results,
halving their duplicate solver work; they do not approximate the domain.
Generation/labeling uses CPU processes and leaves GPU training separate.

Scene simplification and wording choices happen before labeling, once per
premise group. A room setup is omitted only if neither any substantive premise
nor any selected hypothesis mentions room membership. The original distinct-name
conventions remain visible, preserving cardinality. Otherwise its scope remains
explicit. A seeded group-level choice uses either “in the room” wording or
equivalent “room occupant” phrasing, consistently in premises and hypotheses.
Such label-blind choices can still correlate with labels; they are not a proof
that surface shortcuts have been eliminated.

Run `python -m gramforge.fol_nli.surface --dataset OUTPUT --output surface.json`
on the actual exported dataset to measure premise-only, hypothesis-only and
formatting-only logistic probes with three held-out premise-group folds. Use
`--pool POOL` instead for raw generation diagnostics. Reports include balanced
accuracy (chance 1/3), per-label presence/absence and length statistics, and fold
scores. These probes use public input text, not proofs or generation metadata;
above-chance scores need investigation and do not alone establish a causal leak.

The profile rejects identical connective operands and repeated named entities
within a statement. Ground self-relations are also excluded from premise and
hypothesis sampling, including named instantiations of properties. These are
sampling restrictions: they do not assert that sentiment or other relations
are logically irreflexive, and quantified relational witnesses may coincide.
All connective and relation production families remain available. This does
not eliminate every tautology or contradiction obtainable through composition.

`inspect` writes productivity, observed-rule, premise-length and proof-support
reports. A manifest alone does not certify correctness. Only the final audit
establishes which exported rows were successfully rechecked; independent-solver
unknowns are reported separately. Broader construction-specific semantic tests
and training comparisons remain required before calling the successor better.

Nothing is uploaded or published by these commands.

## Verified development checkpoint

The 500-group pilot (`outputs/fol-nli/pilot-v4`) completed in 584 seconds on
16 workers: 3,270 resolved pairs, 220 inconsistent-premise groups and four
unresolved-premise groups. No generation failures occurred. A deliberately
small smoke export used 30 rows, six reserved heuristic examples and support
reference bin one. All 30 passed fresh Vampire and independent Z3 checks with
no unresolved results or disagreements. This validates the plumbing, not the
paper-scale balancing or transfer quality. The larger build must still satisfy
the five-input-bin protocol; the small pilot did not establish that coverage.

The semantic, FOL quality and core grammar/CFG test suites pass 29 tests. A wheel build includes the pipeline and
frozen vocabulary. Historical pilot source snapshots remain separate from the
current mixed/chain-rich sampling profile, which preserves positive weight for
all productions and uses a 25% chain-rich stratum to improve multi-step coverage.
`build --strong-chains` is an experimental stronger weighting of that same
stratum. It keeps all productions positive and leaves mixed-profile derivations
unchanged; the build specification records the option. A paired 1,000-group
development test increased premise-dependent entailments with at least three
cited inputs from 10 to 46, with five-input cases increasing from 1 to 5, but
also had 29 incomplete generations versus zero. These small counts do not
establish production yield or downstream improvement. The audited default
sampling remains unchanged.

Distributed CPU generation uses deterministic index sharding: give workers the
same pool, `--groups`, seed and sources, and distinct `--shard` values with a
shared `--shards`. Initialize the first worker before starting the others. After
all shard completion records exist, run the same build command with `--finalize`
to check that every group is present and write the pool completion record. Worker
count and sharding do not change the indexed generation inputs.

The immutable candidate checkpoints preserve resolved and unresolved solver
outcomes. Fresh runs near a timeout boundary can retain different candidate
subsets on different hardware; exact postprocessing replay uses the frozen pool.
This is recorded provenance, not a claim of bitwise-identical theorem-prover
logs across machines. Source symbols, seed/index derivations and resolved
logical labels remain the reproducible inputs to the audit.
