from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_accessors_register_on_ordinary_import() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    series = frame["a"]
    assert frame.bea.levels()["kind"] == "levels"
    assert series.bea.value_counts(output=True) == {1: 1, 2: 1}


def test_minimal_import_does_not_load_optional_modules() -> None:
    script = """
import json, sys
import pandas as pd
import bea_tools
assert 'pulp' not in sys.modules
assert 'matplotlib' not in sys.modules
r = bea_tools.census(pd.DataFrame({'a': [1, 1]}), ['a'])
json.dumps(r.to_dict(), allow_nan=False)
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(os.environ.get("TMPDIR", "/tmp")),
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"
