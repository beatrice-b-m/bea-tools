# bea-tools 🐝

A Python package of personal data science/analysis focused functions and tools that I use regularly.

## Installation

```bash
pip install bea-tools
```

## Feature hierarchy explorer

The explorer provides independent level counts, a bounded nested census,
observed functional-dependency evidence, sparse pair summaries, and strict JSON
results:

```python
from bea_tools import KeySpec, explore, render_plaintext

result = explore(
    frame,
    dimensions=["site", "study_type"],
    candidate_keys=[KeySpec("exam_side", ("exam_id", "laterality"))],
    top_n=5,
)
print(render_plaintext(result))
```

See the [feature hierarchy explorer guide](docs/feature-hierarchy-explorer.md)
the [hands-on review notebook](examples/feature-hierarchy-explorer.ipynb), and
the [implementation plan](docs/feature-hierarchy-explorer-plan.md).

## Requirements

- Python 3.11+
- pandas >= 3.0
- NumPy >= 1.26 (subject to pandas/Python compatibility)
- pydicom (for DICOM utilities)

Optional features are isolated from the core install:

```bash
pip install 'bea-tools[sampling]'  # PuLP sampler
pip install 'bea-tools[plotting]'  # matplotlib helpers
pip install 'bea-tools[unicode]'   # native-width Unicode rendering
```

## License

MIT
