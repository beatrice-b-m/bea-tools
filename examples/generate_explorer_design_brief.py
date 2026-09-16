"""Regenerate synthetic, executed examples in the standalone design brief.

Run from the repository root: python -m examples.generate_explorer_design_brief
Only the generated section of the Markdown brief is replaced.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

MARKER = "<!-- GENERATED EXAMPLES: DO NOT EDIT BELOW -->"
SETUP = """import json
import pandas as pd
from bea_tools import KeySpec, census, explore, grain, infer_schema, levels, render_plaintext

df = pd.DataFrame({
    "exam_id": ["E1", "E1", "E2", "E2", "E3", "E3"],
    "site": ["North", "North", "South", "South", "South", "South"],
    "side": ["L", "R", "L", "R", "L", "L"],
    "finding": ["clear", "scar", "clear", "clear", None, None],
})"""

# These small cases intentionally cover different outputs without repeating the
# entire combined report for every option. The code is also printed in the brief.
CASES = [
    (
        "Independent values, omitted mass, and canonical topology ordering",
        'result = levels(df, ["site", "finding"], max_levels=2)',
        ("full", "topology"),
        {},
    ),
    (
        "Observed hierarchy: the same result, two presentations",
        'result = census(df, ["site", "side", "finding"])',
        ("full", "topology"),
        {},
    ),
    (
        "Single and composite candidate keys; repeated and singleton support",
        (
            'result = grain(df[["exam_id", "side", "finding"]],\n'
            '               ["exam_id", KeySpec("exam_side", ("exam_id", "side"))])'
        ),
        ("full", "topology"),
        {},
    ),
    (
        "Global and contextual pairs, declared domains, and bounded absence examples",
        (
            "result = explore(\n"
            '    df, ["side", "finding"], include_absence=True,\n'
            '    reference_domains={"finding": ["clear", "scar", None, "nodule"]},\n'
            '    pair_contexts=[{"site": "North"}], max_absence_cells=2,\n'
            ')["sections"]["pairs"]'
        ),
        ("full", "topology"),
        {},
    ),
    (
        "Directional pair classes: reversing feature roles changes the interpretation",
        (
            'mapping = pd.DataFrame({"child": ["a", "b", "c"],\n'
            '                        "parent": ["P", "P", "Q"], "alias": ["x", "y", "z"]})\n'
            'result = explore(mapping, ["child", "parent", "alias"])["sections"]["pairs"]'
        ),
        ("topology",),
        {},
    ),
    (
        "Schema suggestions with reasons and optional dependency evidence",
        (
            "result = infer_schema(\n"
            '    pd.DataFrame({"record_id": [1, 2, 3], "status": ["ready", "ready", "hold"]}),\n'
            '    candidate_keys=["record_id"],\n'
            ")"
        ),
        ("full", "topology"),
        {},
    ),
    (
        "Typed values, literal missing labels, and a warning without quantitative detail",
        (
            'result = levels(pd.DataFrame({"value": pd.Series(\n'
            '    [1, "1", True, "True", None, "<NA>"], dtype=object\n'
            ")}))"
        ),
        ("topology",),
        {},
    ),
    (
        "Conditional pre-selection versus population-preserving post-selection",
        'result = census(df, ["site", "side"], top_n=1, top_n_mode="pre")',
        ("full", "topology"),
        {},
    ),
    (
        "Post-selection retains topology omission notices without omitted quantities",
        'result = census(df, ["site", "side"], top_n=1, top_n_per_parent=True)',
        ("topology",),
        {},
    ),
    (
        "Renderer-only node limit, independent of analytical limits",
        'result = census(df, ["site", "side"])',
        ("topology",),
        {"max_nodes": 0},
    ),
    (
        "Requested pair/context limits remain visible without their sizes",
        (
            "result = explore(\n"
            '    df, ["site", "side", "finding"], max_pairs=1, max_contexts=1,\n'
            '    pair_contexts=[{"exam_id": "E1"}],\n'
            ')["sections"]["pairs"]'
        ),
        ("topology",),
        {},
    ),
    (
        "Undefined evidence after excluding missing values",
        (
            'result = grain(pd.DataFrame({"id": [None], "target": [None]}),\n'
            '               ["id"], dropna=True)'
        ),
        ("topology",),
        {},
    ),
    (
        "Combined report and explicit line-budget truncation",
        'result = explore(df, ["site", "side"], candidate_keys=["exam_id"])',
        ("topology",),
        {"max_lines": 14},
    ),
]


def main() -> None:
    namespace = {}
    exec(SETUP, namespace)  # noqa: S102 - fixed synthetic example, never external input
    blocks = [f"\n**Synthetic setup (shared by the examples)**\n\n```python\n{SETUP}\n```\n"]
    for title, code, modes, overrides in CASES:
        blocks.append(f"\n**{title}**\n\n```python\n{code}\n```\n")
        exec(code, namespace)  # noqa: S102 - code is defined in this module
        for mode in modes:
            options = {"detail": mode, "width": 110, "max_lines": 200, **overrides}
            arguments = ", ".join(f"{key}={value!r}" for key, value in options.items())
            call = f"print(render_plaintext(result, {arguments}))"
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(call, namespace)  # noqa: S102 - call is built from fixed options above
            output = capture.getvalue().rstrip("\n")
            assert len(output.splitlines()) <= options["max_lines"]
            assert all(len(line) <= options["width"] for line in output.splitlines())
            if "max_lines" not in overrides:
                assert "more output not rendered (max_lines)" not in output
            blocks.append(f"\n```python\n{call}\n```\n\n```text\n{output}\n```\n")

    code = """equivalent = pd.DataFrame({
    "exam_id": ["E1", "E1", "E2", "E2"],
    "alias": ["A", "A", "B", "B"],
    "site": ["North", "North", "South", "South"],
})
evidence = grain(equivalent, ["exam_id", "alias"])
print(json.dumps(evidence["targets"][-1], indent=2))"""
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(code, namespace)  # noqa: S102 - code is defined in this module
    blocks.append(
        "\n**An actual structured-result projection: equivalent determinants**\n\n"
        "This is a selected JSON field for illustration, not a topology export API. "
        "Other fields in the same result retain quantitative evidence.\n\n"
        f"```python\n{code}\n```\n\n```json\n{capture.getvalue().rstrip()}\n```\n"
    )
    # Confirm the alternative entry point used in notebooks shares the same result.
    frame = namespace["df"]
    assert frame.bea.levels(["site"]).to_dict() == namespace["levels"](frame, ["site"]).to_dict()
    path = (
        Path(__file__).resolve().parents[1] / "docs" / "feature-hierarchy-explorer-design-brief.md"
    )
    prefix = path.read_text().split(MARKER, 1)[0]
    path.write_text(prefix + MARKER + "\n" + "".join(blocks))
    print(f"Generated {len(CASES)} display scenarios plus a JSON projection: {path}")


if __name__ == "__main__":
    main()
