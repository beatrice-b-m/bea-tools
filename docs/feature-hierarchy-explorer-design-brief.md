Feature hierarchy explorer — conceptual design brief

Please investigate how to present dataset structure most usefully to humans and
language-model agents. Work from this brief and its executed examples; source-code
inspection and implementation are outside this design stage.

**Motivation and intended workflow**

The user often explores raw, semi-unknown data that cannot be exposed directly to
external models. They need both an informative local exploration tool and a way to
manually extract, review, and share selected dataset context. An external model
could then help interpret the layout, formulate questions, or plan analysis without
receiving the source rows or quantitative summaries that must remain internal.

“Topology” here means observed feature values, combinations, nesting and mapping
relationships, candidate entity grain, and exceptions—not spatial topology. The
important questions are: What might a row represent? What varies within an entity?
Which features are nested, equivalent, or cross-cutting? Which relationships change
by subgroup? Which combinations are missing? What should be investigated next?

The desired workflow is **analyze locally → inspect evidence → choose what context
to expose → render and review a shareable description → reason with an external
agent → investigate locally again**. Human interpretability and agent usefulness
both matter; a compact description should preserve relationships and limitations.

**Current analytical capabilities**

| Surface | What it does | Evidence available |
|---|---|---|
| `infer_schema()` | Suggests id/categorical/continuous/unknown roles. | Dtype, cardinality/ratio, missingness, name hints, optional dependency evidence. Heuristic suggestions do not automatically select dimensions or keys. |
| `levels()` | Counts each selected feature independently. | Typed values, counts, shares, reported/omitted levels and row mass, per-feature scope. |
| `census()` | Groups observed prefixes along caller-ordered dimensions. | Parent-linked tree, counts, shares of parent/total, explicit omissions and stopping reasons. |
| `grain()` | Tests caller-supplied single or composite keys (`KeySpec`). | Exact observed dependencies, violating groups, affected rows, singleton/repeated support, undefined results, equivalent/incomparable/coarsest compatible determinants. |
| Pair analysis through `explore()` | Describes dimension pairs globally and under explicit contexts. | Directional `1:1`, `1:n`, `n:1`, `n:m` mappings, Cramér's V, marginals, observed-cell counts, population scopes. |
| Optional absence analysis | Compares observed combinations against empirical or declared domains. | Exact absence totals; zero-support, context-absent, and supported-margin classes; bounded unobserved examples. |
| `explore()` | Combines these perspectives. | Independent levels and census, optional grain and pairs/absence; sections can be inspected separately. |

Functional calls and `DataFrame.bea` accessors provide the same analysis. Results
are versioned, typed, JSON-compatible objects. They include source metadata,
population accounting, statuses, warnings, and conditional-cohort lineage.

**Current display and export choices**

| Content | `detail="full"` (default) | `detail="topology"` |
|---|---|---|
| Column names, typed level labels, missing-value label | Kept | Kept |
| Observed tree paths and dimension order | Kept | Kept |
| Row counts, shares, size/distribution evidence | Quantitative evidence shown where supported by the renderer; shares remain in JSON | Suppressed |
| Frequency-ranked levels and siblings | Frequency order | Canonical typed-value order among included items |
| Dependency outcomes and determinant comparisons | Outcomes plus violations/support | Qualitative outcomes/comparisons only |
| Pair classes and contexts | Kept, with populations and association strength | Kept; Cramér's V and population quantities suppressed |
| Absence domains and examples | Sources, sizes, class totals, examples | Sources and examples retained; sizes and class totals suppressed |
| Schema proposals | Roles, reasons, metrics, optional FD evidence | Roles, dtype/name hints, qualitative FD evidence; metrics suppressed |
| Limits and incompleteness | Quantified omission notices | Nonquantitative omission notices |
| Conditional/status labels | Kept | Kept |
| Warnings | Details shown | Warning codes retained without details; low-retained-fraction warning suppressed |

`render_plaintext()` accepts an entire result or individual section. Trees are
printed parent-first with contiguous subtrees and named dimensions. Display
controls are `detail`, `width`, `max_lines`, `max_nodes`, `missing_label`, and
`unicode_mode` (`safe` ASCII escaping or optional native `display`). Defaults are
full detail, width 100, 200 lines, 1,000 rendered census nodes, and `<NA>` for missing.
Clipping and truncation are explicit. String labels remain distinguishable from
numbers and missing values. Topology mode preserves numeric *labels* and notation
such as `1:n`; it is not a blanket removal of numbers.

Other current views are manually assembled notebook tables and structured JSON.
There is no dedicated graphical/interactive explorer, no automatic label
pseudonymization, and no separate topology-only JSON exporter. Numeric features
are counted as distinct values; dedicated binning, histograms, and continuous
correlation views are not currently provided by this explorer. **The original
result and `to_dict()` still contain quantitative evidence.** When quantities must
stay internal, the new sharing surface is the reviewed rendered string.

Topology mode suppresses size/distribution fields, not all possible inferences.
It intentionally exposes labels, observed paths, contexts, outcomes, and absence
examples; their contents may themselves be sensitive. Visible values/branches can
be counted. Canonical ordering removes displayed frequency ranking, but inclusion
can still reflect `top_n`, `min_count`, or frequency-ordered analytical/renderer
budgets. Qualitative outcomes and heuristic role suggestions also carry evidence.
The user therefore needs deliberate selection and review of what is shared; this
feature is a presentation filter, not a de-identification or privacy guarantee.

**Semantics a useful visualization must preserve**

- Counts describe rows, not automatically unique entities. A census is a chosen
  grouping order, not a discovered or proven natural hierarchy.
- `levels()` uses independent feature populations. Census `top_n_mode="post"`
  retains population counts while suppressing branches; `"pre"` changes the cohort
  and recomputes results. Top-N can be global or per parent.
- In combined pre-mode exploration, pairs use the census cohort; grain uses the
  original input unless `top_n_applies_to="both"`. Derived scopes retain lineage.
- Missing values are included by default. `dropna=True` exclusions are local to
  the relevant analysis. Context requests do not alter the global pair population.
- Exact observed dependencies are sample evidence; singleton keys can satisfy them
  trivially. Determinants are compared on compatible target populations; otherwise
  comparisons are explicitly unavailable. Topology mode hides support strength.
- An unobserved combination is not necessarily impossible. Pairwise support does
  not rule out higher-order restrictions. Neither association nor nesting proves
  causation.
- Analytical controls include selected features/dimensions/keys, confirmed schema
  roles, `top_n`, pre/post mode, per-parent selection, `min_count`, `max_depth`,
  `max_levels`, `max_nodes`, missingness policy, reference domains, context/pair
  limits, absence-example limits, and a low-retained-fraction warning threshold.
  Rendering limits are separate; hidden content is not evidence of absence.
- The structured tree can supply observed paths and their counts within its bounds.
  Pair summaries include marginals and examples, but not a complete joint-frequency
  table or all observed edges. Some proposed visuals require additional analysis.

**Questions for the design exploration**

Propose a small, coherent set of views and workflows for orientation, relationship
discovery, subgroup comparison, exception investigation, and preparation of an
external context description. Explain each view's question, required evidence,
interactions, likely misreadings, and usefulness to humans versus agents.

Distinguish views supportable by current outputs from ideas needing new evidence.
Explore how to show a chosen hierarchy alongside dependency or cross-cutting
relationships; how to remain useful at high cardinality; and how to preserve scope,
incompleteness, and uncertainty when quantitative evidence is deliberately hidden.
Consider what controls would help a user choose structural context to disclose
without assuming that everything available locally belongs in the external brief.

**Executed examples**

The following examples use only synthetic data. They are generated by the actual
installed project code, not handwritten approximations. Each output block is the
complete output of its displayed call; deliberate budget examples contain the
renderer’s own omission markers. Full and topology views use the same result.

Regenerate from the repository root with:
`python -m examples.generate_explorer_design_brief`

<!-- GENERATED EXAMPLES: DO NOT EDIT BELOW -->

**Synthetic setup (shared by the examples)**

```python
import json
import pandas as pd
from bea_tools import KeySpec, census, explore, grain, infer_schema, levels, render_plaintext

df = pd.DataFrame({
    "exam_id": ["E1", "E1", "E2", "E2", "E3", "E3"],
    "site": ["North", "North", "South", "South", "South", "South"],
    "side": ["L", "R", "L", "R", "L", "L"],
    "finding": ["clear", "scar", "clear", "clear", None, None],
})
```

**Independent values, omitted mass, and canonical topology ordering**

```python
result = levels(df, ["site", "finding"], max_levels=2)
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Levels (computed)
  site: 2/2 levels; 6/6 evaluated rows reported
    'South': 4 rows
    'North': 2 rows
  finding: 2/3 levels; 5/6 evaluated rows reported
    'clear': 3 rows
    <NA>: 2 rows
    ... 1 level / 1 row not reported
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Levels (computed)
  site
    'North'
    'South'
  finding
    'clear'
    <NA>
    ... additional levels not reported (analysis limits)
```

**Observed hierarchy: the same result, two presentations**

```python
result = census(df, ["site", "side", "finding"])
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Census (computed)
  rows: 6 evaluated / 6 input; excluded: 0 missing, 0 restricted
  path: site > side > finding
  total: 6 rows
  site='South': 4 rows
    side='L': 3 rows
      finding=<NA>: 2 rows
      finding='clear': 1 row
    side='R': 1 row
      finding='clear': 1 row
  site='North': 2 rows
    side='L': 1 row
      finding='clear': 1 row
    side='R': 1 row
      finding='scar': 1 row
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Census (computed)
  path: site > side > finding
  site='North'
    side='L'
      finding='clear'
    side='R'
      finding='scar'
  site='South'
    side='L'
      finding='clear'
      finding=<NA>
    side='R'
      finding='clear'
```

**Single and composite candidate keys; repeated and singleton support**

```python
result = grain(df[["exam_id", "side", "finding"]],
               ["exam_id", KeySpec("exam_side", ("exam_id", "side"))])
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Grain (computed)
  Observed dependencies; sample evidence does not establish semantic grain.
  exam_id [exam_id] -> side
    observed dependency fails; 2/3 violating groups; 4 affected rows
    support: 6 rows; 0 singleton, 3 repeated groups
  exam_id [exam_id] -> finding
    observed dependency fails; 1/3 violating groups; 2 affected rows
    support: 6 rows; 0 singleton, 3 repeated groups
  exam_side [exam_id, side] -> finding
    observed dependency holds; 0/5 violating groups; 0 affected rows
    support: 6 rows; 4 singleton, 1 repeated groups
  side: grain undetermined by supplied keys
  finding: coarsest observed candidates: exam_side
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Grain (computed)
  Observed dependencies; sample evidence does not establish semantic grain.
  exam_id [exam_id] -> side
    observed dependency fails
  exam_id [exam_id] -> finding
    observed dependency fails
  exam_side [exam_id, side] -> finding
    observed dependency holds
  side: grain undetermined by supplied keys
  finding: coarsest observed candidates: exam_side
```

**Global and contextual pairs, declared domains, and bounded absence examples**

```python
result = explore(
    df, ["side", "finding"], include_absence=True,
    reference_domains={"finding": ["clear", "scar", None, "nodule"]},
    pair_contexts=[{"site": "North"}], max_absence_cells=2,
)["sections"]["pairs"]
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Pairs (computed)
  A / B is left / right; relations describe observed pairs only.
  Pair support does not rule out higher-order constraints.
  side / finding [global]
    rows: 6 evaluated / 6 input; excluded: 0 missing, 0 restricted
    observed relation: n:m (many-to-many)
    Cramer's V: 0.707107
    unobserved: 4/8 domain cells (not evidence of impossibility)
    domain side: 2 levels (empirical observed)
    domain finding: 4 levels (caller declared)
    zero support in cohort: 2 cells
    level absent in context: 0 cells
    within supported margins: 2 cells
      unobserved example: 'L' / 'nodule'
      unobserved example: 'L' / 'scar'
    ... 2 unobserved examples not reported
  side / finding [site='North']
    rows: 2 evaluated / 6 input; excluded: 0 missing, 4 restricted
    observed relation: 1:1 (one-to-one)
    Cramer's V: 1
    unobserved: 6/8 domain cells (not evidence of impossibility)
    domain side: 2 levels (empirical observed)
    domain finding: 4 levels (caller declared)
    zero support in cohort: 2 cells
    level absent in context: 2 cells
    within supported margins: 2 cells
    ... 6 unobserved examples not reported
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Pairs (computed)
  A / B is left / right; relations describe observed pairs only.
  Pair support does not rule out higher-order constraints.
  side / finding [global]
    observed relation: n:m (many-to-many)
    unobserved combinations (not evidence of impossibility)
    domain side: empirical observed
    domain finding: caller declared
      unobserved example: 'L' / 'nodule'
      unobserved example: 'L' / 'scar'
    ... additional unobserved examples not reported
  side / finding [site='North']
    observed relation: 1:1 (one-to-one)
    unobserved combinations (not evidence of impossibility)
    domain side: empirical observed
    domain finding: caller declared
    ... additional unobserved examples not reported
```

**Directional pair classes: reversing feature roles changes the interpretation**

```python
mapping = pd.DataFrame({"child": ["a", "b", "c"],
                        "parent": ["P", "P", "Q"], "alias": ["x", "y", "z"]})
result = explore(mapping, ["child", "parent", "alias"])["sections"]["pairs"]
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Pairs (computed)
  A / B is left / right; relations describe observed pairs only.
  Pair support does not rule out higher-order constraints.
  child / parent [global]
    observed relation: n:1 (many A to one B)
  child / alias [global]
    observed relation: 1:1 (one-to-one)
  parent / alias [global]
    observed relation: 1:n (one A to many B)
```

**Schema suggestions with reasons and optional dependency evidence**

```python
result = infer_schema(
    pd.DataFrame({"record_id": [1, 2, 3], "status": ["ready", "ready", "hold"]}),
    candidate_keys=["record_id"],
)
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Schema proposals (suggestions) (computed)
  record_id: suggested id
    cardinality: 3
    cardinality ratio: 1.0
    dtype: int64
    missing rows: 0
    name hint id: True
    dependency evidence: evaluated
  status: suggested categorical
    cardinality: 2
    cardinality ratio: 0.6666666666666666
    dtype: str
    missing rows: 0
    dependency evidence: evaluated
      record_id: holds=True, 3 groups
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Schema proposals (suggestions) (computed)
  record_id: suggested id
    dtype: int64
    name hint id: True
    dependency evidence: evaluated
  status: suggested categorical
    dtype: str
    dependency evidence: evaluated
      record_id: holds=True
```

**Typed values, literal missing labels, and a warning without quantitative detail**

```python
result = levels(pd.DataFrame({"value": pd.Series(
    [1, "1", True, "True", None, "<NA>"], dtype=object
)}))
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Levels (computed)
  Warning: MIXED_LEVEL_TYPES
  value
    True
    1
    '1'
    '<NA>'
    'True'
    <NA>
```

**Conditional pre-selection versus population-preserving post-selection**

```python
result = census(df, ["site", "side"], top_n=1, top_n_mode="pre")
```

```python
print(render_plaintext(result, detail='full', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Census (computed, conditional)
  rows: 3 evaluated / 6 input; excluded: 0 missing, 3 restricted
  path: site > side
  total: 3 rows
  site='South': 3 rows
    side='L': 3 rows
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Census (computed, conditional)
  path: site > side
  site='South'
    side='L'
```

**Post-selection retains topology omission notices without omitted quantities**

```python
result = census(df, ["site", "side"], top_n=1, top_n_per_parent=True)
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Census (partial)
  path: site > side
    ... child branches omitted (top_n)
  site='South'
    ... child branches omitted (top_n)
    side='L'
```

**Renderer-only node limit, independent of analytical limits**

```python
result = census(df, ["site", "side"])
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200, max_nodes=0))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Census (computed)
  path: site > side
  ... additional nodes not rendered (renderer max_nodes)
```

**Requested pair/context limits remain visible without their sizes**

```python
result = explore(
    df, ["site", "side", "finding"], max_pairs=1, max_contexts=1,
    pair_contexts=[{"exam_id": "E1"}],
)["sections"]["pairs"]
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Pairs (computed)
  ... requested pairs or contexts omitted (output limits)
  A / B is left / right; relations describe observed pairs only.
  Pair support does not rule out higher-order constraints.
  site / side [global]
    observed relation: n:m (many-to-many)
```

**Undefined evidence after excluding missing values**

```python
result = grain(pd.DataFrame({"id": [None], "target": [None]}),
               ["id"], dropna=True)
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=200))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Grain (computed)
  Observed dependencies; sample evidence does not establish semantic grain.
  id [id] -> target
    observed dependency undefined (no evaluated groups)
  target: grain undetermined by supplied keys
```

**Combined report and explicit line-budget truncation**

```python
result = explore(df, ["site", "side"], candidate_keys=["exam_id"])
```

```python
print(render_plaintext(result, detail='topology', width=110, max_lines=14))
```

```text
bea-tools feature explorer v0.2
Topology display (quantitative evidence suppressed)
Levels (computed)
  site
    'North'
    'South'
  side
    'L'
    'R'
Census (computed)
  path: site > side
  site='North'
    side='L'
... more output not rendered (max_lines)
```

**An actual structured-result projection: equivalent determinants**

This is a selected JSON field for illustration, not a topology export API. Other fields in the same result retain quantitative evidence.

```python
equivalent = pd.DataFrame({
    "exam_id": ["E1", "E1", "E2", "E2"],
    "alias": ["A", "A", "B", "B"],
    "site": ["North", "North", "South", "South"],
})
evidence = grain(equivalent, ["exam_id", "alias"])
print(json.dumps(evidence["targets"][-1], indent=2))
```

```json
{
  "target": {
    "type": "string",
    "value": "site"
  },
  "determining_keys": [
    "exam_id",
    "alias"
  ],
  "assignment": "compatible",
  "cross_key_comparison": "comparable",
  "coarsest_candidates": [
    "exam_id",
    "alias"
  ],
  "equivalent_determinants": [
    [
      "exam_id",
      "alias"
    ]
  ],
  "incomparable_candidates": []
}
```
