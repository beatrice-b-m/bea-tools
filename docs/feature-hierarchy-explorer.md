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
print(render_plaintext(combined, width=80, max_lines=40))
```

An omitted combination is only unobserved in the stated scope. Output limits
are not evidence that omitted combinations are absent, pairwise saturation does
not rule out higher-order constraints, and an `undetermined` grain assignment
does not prove that the real-world grain is finer.

`DataFrame.bea.levels`, `.census`, `.grain`, `.explore`, and `.infer_schema`
delegate to the same functions after an ordinary `import bea_tools`.
