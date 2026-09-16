# Feature hierarchy explorer

The explorer separates four questions that are easy to conflate:

- `levels()` counts each selected feature on its own population.
- `census()` counts observed prefixes in an explicitly ordered path.
- `grain()` tests exact functional dependencies for keys you supply.
- `explore()` combines those operations and can add pair/absence summaries.

All outputs use schema version `0.2`, contain only standard-library data, and
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

Safe mode escapes non-ASCII and terminal controls. Native Unicode display requires
`bea-tools[unicode]` and clips by terminal cell width. Programmatic consumers should
use typed JSON fields rather than parsing this presentation format.
