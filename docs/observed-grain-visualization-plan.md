# Observed-grain visualization implementation plan

## Deliverables

1. Add an explicit common-scope graph summary to `grain()` and the grain section
   of `explore()`. Retain the existing target-specific evidence. Compare all
   supplied keys, merge observationally equivalent groupings, preserve multiple
   parents, and transitively reduce the drawing. Assign features to all of their
   coarsest supported nodes; keep unplaced features explicit.
2. Provide static SVG and standalone interactive HTML renderers. The primary
   grain view uses layered, readable cards with coarse-to-fine arrows. Include a
   key/feature matrix, selectable feature evidence, attribute collapse, key focus,
   and an optional failed-dependency overlay. Use deterministic native SVG layout
   so exporting does not require a system Graphviz installation.
3. Render levels as frequency bars (typed lists in topology mode), census as an
   aligned-bar tree (uniform topology tree), and pairs as directional mapping
   matrices with a separate full-access association view. Keep contexts separate.
4. Add on-demand selected-pair joint counts and a corresponding heatmap, without
   expanding ordinary pair results into joint tables.
5. Document APIs and examples; verify semantics, scope isolation, quantitative
   filtering, escaping, deterministic geometry, import behavior, and rendered
   figures. Track changes in scoped commits.

## Evidence and disclosure rules

- With `dropna=False`, the graph uses every input row and treats missing values
  as levels. With `dropna=True`, it uses rows complete across all candidate-key
  components. A target with further missing-value exclusions is shown in the
  evidence table but remains unplaced because its population differs.
- Empty populations do not establish equivalence or refinement. Composite keys
  remain atomic. Only exact relationships enter the hierarchy.
- Feature placement uses common-scope evidence, never a target-specific
  equivalence promoted to a global merge. Key-component features are represented
  by their key headings rather than duplicated as attributes.
- Topology presentation is an allowlisted projection constructed before SVG or
  HTML serialization. No raw analytical payload, frequencies, quantitative
  support, ranks, or quantity-driven styling is embedded. Canonical ordering
  replaces frequency ordering; explicit qualitative omissions remain.
- Layout reflects structural relationships and readable content, not group sizes.
  Full census and level bars retain their original denominators and omitted mass.
- Contexts and section populations remain labeled and separate. These are maps
  of observed groupings, not declarations of semantic entities.

## Acceptance

Exercise chains, diamonds, equivalent/composite keys, incomparable determinants,
empty inputs, mismatched missingness, conditional cohorts, mixed typed labels,
long/hostile labels, and omission budgets. Inspect generated SVG/HTML and run the
existing explorer and package regression suites. Optional icicles and automatic
entity discovery are outside this implementation; aligned bars supply the
requested quantitative census view.
