# bea-tools 🐝

A Python package of personal data science/analysis focused functions and tools that I use regularly.

## Installation

Install the latest published release from PyPI:

```bash
python -m pip install bea-tools
```

Optional features are available as extras:

```bash
python -m pip install 'bea-tools[sampling]'  # PuLP sampler
python -m pip install 'bea-tools[plotting]'  # matplotlib helpers
python -m pip install 'bea-tools[unicode]'   # native-width Unicode rendering
```

Multiple extras can be installed together, for example:

```bash
python -m pip install 'bea-tools[sampling,plotting,unicode]'
```

## Development installation

The repository uses [uv](https://docs.astral.sh/uv/) to create a locked
development environment. After installing uv, clone the repository and sync
the project with its development tools and optional dependencies:

```bash
git clone https://github.com/beatrice-b-m/bea-tools.git
cd bea-tools
uv sync --group dev --extra sampling --extra plotting --extra unicode
```

`uv sync` creates (or updates) `.venv` and installs the checked-out package in
editable mode. Source-code changes are therefore available to commands run
with `uv run` without reinstalling the package.

Run the test suite and linter from the repository root:

```bash
uv run --group dev pytest bea_tools/testing -q
uv run --group dev ruff check bea_tools/_explore bea_tools/testing/explore
```

CI checks Python 3.11–3.13, optional dependencies, and a separate environment
with the minimum supported runtime dependencies. The minimum-version job also
installs pytest and Hypothesis so it runs the differential tests as well as the
example-based tests. Hypothesis is a development dependency, not a runtime
requirement.

### Install the checkout into an existing Python environment

To use the in-development checkout as a package in an existing virtual or
Conda environment, activate that environment and run this from the repository
root:

```bash
python -m pip install --editable .
```

To include optional features, specify one or more extras on the editable
requirement:

```bash
python -m pip install --editable '.[sampling,plotting,unicode]'
```

Using `python -m pip` ensures the package is installed into the environment
owned by that `python`. If the environment is not activated, uv can target its
interpreter explicitly:

```bash
uv pip install --python /path/to/environment/bin/python --editable .
```

Confirm which checkout is being imported with:

```bash
python -c "import bea_tools; print(bea_tools.__file__)"
```

## Feature hierarchy explorer

The explorer provides independent level counts, a bounded nested census,
observed functional-dependency evidence, sparse pair summaries, and strict JSON
results:

```python
from bea_tools import KeySpec, explore, render_plaintext, render_svg, render_html

result = explore(
    frame,
    dimensions=["site", "study_type"],
    candidate_keys=[KeySpec("exam_side", ("exam_id", "laterality"))],
    top_n=5,
)
print(render_plaintext(result))

# Primary graphical output: observed grain cards and feature placements.
from pathlib import Path
Path("grain.svg").write_text(render_svg(result), encoding="utf-8")
Path("grain.html").write_text(render_html(result), encoding="utf-8")

# Share layout and qualitative relationships without counts or distributions.
external_context = render_plaintext(result, detail="topology")
```

SVG and interactive HTML also support levels, census, and pair matrices through
`section=`. Both provide full and topology-only exports without extra dependencies.
See [graphical output usage](docs/feature-hierarchy-explorer.md#graphical-outputs)
and the [executable gallery](examples/observed_grain_graph.py).

See the [feature hierarchy explorer guide](docs/feature-hierarchy-explorer.md),
the [hands-on review notebook](examples/feature-hierarchy-explorer.ipynb), and
the [implementation plan](docs/feature-hierarchy-explorer-plan.md).

The notebook includes executed SVG figures, interactive HTML previews, and an
editable graphical playground covering all four new presentation APIs. Run it
from the repository root with:

```bash
uv run --group dev --with jupyterlab jupyter lab examples/feature-hierarchy-explorer.ipynb
```

Run all cells first, then edit the playground's section, keys, disclosure mode,
view, and output limits. Interactive previews also save standalone HTML files in
a fresh temporary output directory if the notebook viewer blocks JavaScript.

## Requirements

- Python 3.11+
- pandas >= 3.0
- NumPy >= 1.26 (subject to pandas/Python compatibility)
- pydicom (for DICOM utilities)

## License

MIT
