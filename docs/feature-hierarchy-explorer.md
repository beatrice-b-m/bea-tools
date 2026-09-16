# Feature hierarchy explorer

The explorer separates four questions that are easy to conflate:

- `levels()` counts each selected feature on its own population.
- `census()` counts observed prefixes in an explicitly ordered path.
- `grain()` tests exact functional dependencies for keys you supply.
- `explore()` combines those operations and can add pair/absence summaries.

All outputs use schema version `0.3`, contain only standard-library data, and
can be passed to `json.dumps(result.to_dict(), allow_nan=False)`. Values and
column labels are typed, so booleans, integers, floats, strings, and missing
values do not collapse into one another.

## Examples

```python
import pandas as pd
from bea_tools import KeySpec, census, explore, grain, levels, render_plaintext

df = pd.DataFrame({
    "exam_id": [1, 1, 2, 2],
    "side": ["L", "R", "L", "R"],
    "finding": ["clear", "scar", "clear", "clear"],
})

# Independent top-five counts; no conjunctive row filtering.
independent = levels(df, ["side", "finding"], top_n=5)

# Population-preserving bounded expansion. Exact top-N still scans the data.
nested = census(df, ["side", "finding"], top_n=5, max_nodes=100)

# Conditional pre filtering explicitly changes the evaluated population.
conditional = census(df, ["side", "finding"], top_n=1, top_n_mode="pre")

# Composite keys are explicit, so tuple-labelled columns remain unambiguous.
evidence = grain(df, [KeySpec("exam_side", ("exam_id", "side"))])

combined = explore(
    df,
    dimensions=["side", "finding"],
    candidate_keys=["exam_id"],
    reference_domains={"side": ["L", "R", "U"]},
    include_absence=True,
)
print(render_plaintext(combined, width=100, max_lines=120))
```

An omitted combination is only unobserved in the stated scope. Output limits
are not evidence that omitted combinations are absent, pairwise saturation does
not rule out higher-order constraints, and an `undetermined` grain assignment
does not prove that the real-world grain is finer.

`DataFrame.bea.levels`, `.census`, `.grain`, `.explore`, and `.infer_schema`
delegate to the same functions after an ordinary `import bea_tools`.

## Inspecting results without looking up codes

Printing a result, evaluating it interactively, or displaying it in a notebook
now shows the plaintext view with actual column names and level values:

```python
result = census(df, ["side", "finding"])
print(result)  # e.g. side='L', followed by finding='clear'
```

This default display is bounded to 100 characters per line, 40 lines, and 100
census nodes, with explicit truncation notices. It includes quantitative evidence.
Use `render_plaintext(result, ...)` to change the display budgets or request
`detail="topology"`. `joint_counts()` also supports plaintext display, including
named columns, context predicates, and observed cell values.

The default `result.to_dict()` and mapping access remain compact: census nodes
refer to `features` and `level_dictionary`, and graph links retain their IDs.
For structured inspection without manual lookups, request a resolved export:

```python
readable = result.to_dict(resolve_references=True)
node = readable["tree"]["nodes"][0]
print(node["label"])         # side='L'
print(node["column"])        # {'type': 'string', 'value': 'side'}
print(node["value"])         # {'type': 'string', 'value': 'L'}
print(node["parent_label"])  # All evaluated rows
```

Resolved exports preserve all original IDs, counts, ordering, and typed values;
they add local names and values as follows:

| Result | Added information |
| --- | --- |
| Census | Node `column`, `value`, `label`, and `parent_label`; tree `dimension_columns` and `dimension_labels`; named dictionary entries |
| Pre-filter census selections | Retained-set `column`, `column_label`, typed `values`, predicate `labels`, and per-parent `path_values` predicates |
| Levels | Feature `column_label`; each level's `column` and predicate `label` |
| Grain | Key `column_labels`, evidence/assignment `target_label`, edge `source_keys`/`target_keys`, and assignment `node_keys` |
| Pairs | `column_labels`, labeled context predicates, and absence examples with `a_column`, `b_column`, and `label` |
| Joint counts | `column_labels`, labeled contexts, and cells with `a_column`, `b_column`, `a_value`, `b_value`, and `label`; `a`/`b` remain indexes |
| Schema proposals and warnings | `column_label`, with typed warning `column` values |

`explore()` resolves each computed section using its own feature mappings.
Unrequested sections remain unrequested. The resolved dictionary is an independent
copy, so editing it does not modify the result. Resolving requires no dataframe or
recomputation, but repeated names and values increase export size. Display labels
are for reading; typed fields distinguish values such as `1`, `1.0`, `True`,
`'1'`, missingness, and the literal string `'<NA>'`. IDs remain necessary because
the same level can occur under different parents. A `parent_label` names the
immediate parent; use parent IDs to reconstruct the complete path.

Feature warnings now carry their typed column and display its name. Census
dictionaries include values referenced by pre-filter selections even when output
budgets omit those nodes or another dimension's filter excludes those values.
Older saved results that lack these dictionary entries must be recomputed before
resolving their retained sets; resolution raises an actionable error in that case.
These are additive changes to schema `0.3`.

`visualization_data()` also includes labels beside references: census rows have
`parent_label`; grain edges have `source_label`/`target_label`, features have
`node_labels`, nodes have `attribute_labels`, and evidence has `feature_label`.
Pair and joint cells include `a_label`/`b_label`. Structural IDs such as `c0`
and `g0` remain available for linking, while readers can use the adjacent labels.
These additions apply to both full and topology projections and preserve topology
disclosure filtering. Resolved analytical exports are **not** disclosure-filtered;
use `visualization_data(..., detail="topology")` or a topology renderer when
quantities must be omitted.

## Population scopes

Pair contexts are evaluated independently. With `dropna=True`, the unconditioned
pair excludes missing values only in its two dimensions. A contextual pair also
excludes missing values in that context's columns; requesting another context
does not change either population. Absence domains and global support are based
on the unconditioned pair population.

In pre mode, pair analysis uses the census cohort. Grain uses that cohort only
with `top_n_applies_to="both"`. These sections retain `scope_metadata` containing
`source_scope`, `conditional`, and the parent `scope`. Each derived scope accounts
for the original input rows, accumulates missing and restriction exclusions, and
includes the census scope in its lineage. Section `source` describes the immediate
input dataframe, which may be the cohort. Exclusions are disjoint: missingness is
counted at each stage only among rows surviving earlier stages.

Grain compares determinant relationships on the same rows used for the target's
dependencies. When the candidate keys have different target populations, the
comparison remains `not_comparable`; it does not infer equivalence or ordering
from dependencies evaluated on other populations.

## Reading the plaintext view

`render_plaintext()` accepts an explorer result or its dictionary, including an
individual section of `explore()`. Census nodes are displayed in depth-first order
using their parent links. Each line names its dimension, and siblings retain the
producer's ranking. The underlying JSON stays in its deterministic breadth-first
budget order. For example:

```text
  total: 3 rows
  site='North': 2 rows
    modality='CT': 1 row
    modality='MRI': 1 row
  site='South': 1 row
    modality='CT': 1 row
```

The view includes:

- Evaluated/input row counts, missing and restriction exclusions, and conditional
  cohort labels. Root totals remain visible for a census with `max_nodes=0`.
- Omitted child rows and levels with their stopping reasons; omitted level-count
  mass; and requested pairs/contexts excluded by their budgets.
- Quoted, escaped string values, so `1`, `'1'`, `True`, `'True'`, `<NA>`, and
  `'<NA>'` stay distinct. Ordinary identifier column names remain unquoted.
- Named pairs and context predicates, relation orientation, and explicit reasons
  for undefined associations. Cramér's V is displayed with six significant digits;
  JSON retains the numeric result.
- Absence domain sources and sizes, separate zero-support/context-absent/supported-
  margin counts, and bounded examples. These describe unobserved cells, not
  impossible combinations.
- Observed dependency truth, violating/evaluated groups, affected rows, singleton
  and repeated-group support, undefined reasons, and determinant comparisons.
- Schema proposals with their reasons and dependency evidence, plus analysis warnings.

`width`, `max_lines`, and renderer `max_nodes` are finite integer display budgets.
The renderer's node budget selects an ancestor-closed prefix of the census's node
list, then prints those nodes by subtree. It does not change the analytical result.
Long lines end with `...`; a final `max_lines` marker appears only when additional
content exists, within the requested line budget. A separate `max_nodes` notice
identifies nodes hidden only by the renderer. Increase the display budgets or
render one section to inspect more detail:

```python
print(render_plaintext(combined["sections"]["pairs"], width=100, max_lines=100))
```

### Topology-only display

Use `detail="topology"` when the recipient needs the dataset layout but should
not receive its size or distribution evidence:

```python
external_context = render_plaintext(
    combined,
    detail="topology",
    width=100,
    max_lines=200,
)
print(external_context)
```

This mode retains column and level labels, observed hierarchy paths, qualitative
functional-dependency results, pair relation classes, contexts, declared-domain
sources, and bounded examples of unobserved combinations. It suppresses:

- row, level, node, pair, context, group, cell, and domain sizes;
- shares, missing/excluded population totals, affected-row and support totals;
- Cramér's V, cardinality ratios, missing-row counts, and other schema metrics;
- quantitative omission markers and low-retained-fraction warnings.

Values ranked by frequency in the analytical result are sorted canonically for
display, so their printed order does not reveal relative prevalence. Omission
markers remain, without quantities, so the recipient can distinguish a complete
display from one limited by `top_n`, `max_levels`, `max_nodes`, `max_lines`, pair,
context, or absence-example budgets.

Topology mode is a presentation filter, not de-identification. It intentionally
reveals column names, categorical values, observed paths, context values,
dependency outcomes, pair relation classes, and unobserved examples. A result
created with frequency-based selection such as `top_n` can also reveal that its
included values passed that selection, even though their display order is
canonical. Review those labels and choices before sending the text outside your
environment. The original `ExplorerResult` still contains all quantitative data;
share only the rendered string when those fields must remain internal.

Safe mode escapes non-ASCII and terminal controls. Native Unicode display requires
`bea-tools[unicode]` and clips by terminal cell width. Programmatic consumers should
use typed JSON fields rather than parsing this presentation format.

For a standalone conceptual overview with executed full/topology examples, see
[the visualization design brief](feature-hierarchy-explorer-design-brief.md).

## Common-scope observed-grain graph

Schema `0.3` adds `grain_result["graph"]`; the existing `dependencies` and
`targets` retain their target-specific populations and semantics. The graph has
its own `scope`, `missingness`, complete directed `key_relationships`, support
and dependency records, merged `nodes`, reduced coarse-to-fine `edges`, feature
`assignments`, and explicit `unplaced` features.

With `dropna=False`, every input row participates and missing values are levels.
With `dropna=True`, the graph uses rows complete across every supplied key
component. Target dependencies are evaluated within that population; if a target
has further missing-value exclusions, its evidence has `scope_compatible=False`
and a separate scope. That target is not assigned to the graph. Empty populations
do not establish key equivalence, even for keys with identical components.

Equivalent keys retain every supplied name in one node. Edges mean **finer
grouping**, the reverse of a functional-dependency arrow. Only exact refinements
enter the graph. Redundant skip-level edges are removed from the drawing while
all tested key relationships remain available. Cross-cutting keys can share
children. Features with multiple incomparable coarsest determinants are assigned
to every such node; unsupported features are “Not placed by tested keys,” not
assumed to be row-level. Key components are represented by their key headings.

The graph is based only on supplied candidates. To include a categorical grouping
such as site, include it in `candidate_keys`. Composite keys use `KeySpec` and
remain atomic. Adding a candidate with missing values can change the common
population; always inspect the graph scope. Conditional `explore()` grain
sections carry the original population accounting and cohort lineage into it.

## Graphical outputs

For a runnable tour, open the
[hands-on notebook](../examples/feature-hierarchy-explorer.ipynb) and run all cells.
It contains saved full/topology SVG figures, isolated interactive HTML previews,
and examples of equivalent keys, shared assignments, multiple parents, missingness
scopes, contextual pair matrices, and joint-count budget guards. Section 10 is an
editable playground for `render_svg()`, `render_html()`, and `visualization_data()`;
the selected-pair section demonstrates `joint_counts()` separately. Every HTML
preview is also exported to the temporary directory printed during setup.
The notebook contains unrestricted evidence; share an individual topology export
rather than the whole notebook when counts must remain internal.

`render_svg()` produces a standalone static figure; `render_html()` produces a
standalone interactive document. Both accept an `ExplorerResult` or its dictionary
and require no optional packages, Graphviz executable, remote assets, or running
server. They consume the result's evidence without accessing the dataframe.
Combined `explore()` results default to the grain map; use `section="levels"`,
`"census"`, or `"pairs"` for another section. An unrequested section raises an
actionable error instead of inventing evidence.

```python
from pathlib import Path
from bea_tools import KeySpec, grain, render_svg, render_html, visualization_data

result = grain(df, ["exam_id", KeySpec("exam_side", ("exam_id", "side"))])
Path("grain.svg").write_text(render_svg(result), encoding="utf-8")
Path("grain.html").write_text(render_html(result), encoding="utf-8")

# Inline notebook figure:
from IPython.display import SVG, display
display(SVG(render_svg(result)))

# Explicitly filtered structure for an agent or another rendering client:
structural_summary = visualization_data(result, detail="topology")
```

The native SVG layout orders candidates by structural depth and input key order,
with content-sized cards and coarse-to-fine arrows. Counts never control card area
or edge thickness. Equivalent keys appear together, composite components remain
in one heading, shared features are labeled, and unplaced features stay visible.
Long labels wrap. Dense or wide graphs can be scrolled in the HTML view; use the
matrix or focus controls when the map is crowded. Layout is deterministic rather
than optimized for minimum edge crossings.

The HTML grain view supports:

- Selecting a feature in a card, matrix, or menu to highlight its assignments and
  open the candidate-by-candidate evidence table. Key components can be selected
  through the menu or matrix. Full tables show violations, singleton/repeated
  support, affected rows, and population accounting.
- Switching between map and candidate-key × target-feature matrix. Matrix cells
  distinguish constant, varying, undefined, and untested evidence. An asterisk
  marks a different target population; those cells do not justify map placement.
- Hiding attribute lists, focusing a key and its immediate neighbors while other
  connections stay dimly visible, and showing varying features as an optional
  exception overlay. Exceptions never change the exact hierarchy.

For static variants, use `render_svg(result, view="matrix")` or
`render_svg(result, show_exceptions=True)`.

| Section | Full view | Topology view |
| --- | --- | --- |
| Grain | Layered cards, support, selectable evidence | Same groupings and placement; qualitative evidence |
| Levels | Frequency bars, explicit missingness, omitted mass | Canonically ordered typed labels |
| Census | Expandable HTML tree and static aligned bars | Uniform paths with qualitative omissions |
| Pairs | Directional mapping matrix or separate Cramér's V layer | Mapping classes only |

Level bars use each feature's evaluated population. Census bars use the root
population, and HTML also reports each branch's share of its parent. Omitted
levels/branches have their own mass rather than redistributing it to visible
entries. HTML census branches collapse through the arrow buttons; the static
figure remains available underneath. Missing values are labeled `<NA>` when
included, and missing exclusions appear in the full population captions.

Pair cells describe row-feature → column-feature mapping, with `1:n` reversed to
`n:1` in the opposite direction. `view="association"` selects Cramér's V for
static SVG; HTML supplies an encoding selector. Association is unavailable in
topology mode. Global and contextual matrices retain the same feature order and
are drawn separately. The context selector keeps the global comparison visible;
cell titles explain each population. Analysis budget omissions remain explicit,
and untested pairs are not treated as undefined or as observed relationships.
An unconditioned matrix can itself use a conditional census cohort: its cell
population captions retain that distinction.

### Selected-pair joint cells

Complete joint frequencies are computed only when requested directly:

```python
from bea_tools import joint_counts, render_svg

cells = joint_counts(
    df, ["side", "finding"],
    # context={"site": "North"},  # optional; disjoint from the selected pair
    dropna=False,
    max_cells=2500,
)
Path("joint.svg").write_text(render_svg(cells), encoding="utf-8")
```

`joint_counts()` returns canonically ordered typed `a` and `b` domains, sparse
observed `cells` with domain indexes and counts, context predicates, and a scope.
Missingness is evaluated across the selected pair and context columns before
context restriction. `max_cells` bounds the full supported-domain product,
including blank heatmap cells; exceeding it raises rather than dropping mass.
Choose a narrower context or explicitly increase the budget. Full heatmaps use
cell counts; topology heatmaps show uniform observed-cell marks. Blank cells mean
unobserved within the displayed scope, not impossible combinations.

### Disclosure and compatibility

All graphic renderers first call `visualization_data()`, an allowlisted projection.
Topology exports contain no raw analytical JSON, counts, shares, support metrics,
association strengths, frequency ranks, quantity-controlled styling, or hidden
quantitative tooltips. This applies to the entire HTML file, including inactive
views and evidence tables. Frequency-ranked levels and census siblings are
canonically reordered, and presentation IDs are assigned after that ordering.
The compact projection also provides structural evidence to text/agent consumers;
full projections retain quantitative evidence independently of audience.

Topology is still a presentation policy, not de-identification: names, values,
observed structure, qualitative failures, and the effects of analytical selection
remain visible. Share the rendered export or the reviewed projection, not
`result.to_dict()`, when quantities must remain internal. The renderers do not
mutate the analytical result. Missing graph evidence in older saved results
requires recomputing `grain()`; target-specific equivalences are never used as a
fallback global graph.

Run the executable gallery example to generate all supported surfaces in both
modes, plus the grain matrix and exception figure:

```bash
python examples/observed_grain_graph.py /tmp/bea-explorer-figures
```

The [implementation plan](observed-grain-visualization-plan.md) records the scope
and evidence rules. Optional icicle layouts and automatic entity/composite-key
discovery are not included.
