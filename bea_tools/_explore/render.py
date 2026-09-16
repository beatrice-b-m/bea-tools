"""Bounded plaintext rendering from explorer results only."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .encoding import ScalarIdentity, display_scalar, validate_limit
from .result import ExplorerResult

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def _identity(record: Mapping[str, Any]) -> ScalarIdentity:
    kind = record["type"]
    if kind == "tuple":
        value = tuple(_identity(item) for item in record["value"])
    else:
        value = record.get("value")
    metadata = tuple(
        sorted((key, str(value)) for key, value in record.items() if key not in {"type", "value"})
    )
    return ScalarIdentity(kind, value, metadata)


def _safe(text: str, unicode_mode: str) -> str:
    text = _CONTROL.sub(lambda match: f"\\u{ord(match.group()):04x}", text)
    if unicode_mode == "safe":
        return text.encode("ascii", "backslashreplace").decode("ascii")
    if unicode_mode != "display":
        raise ValueError("unicode_mode must be 'safe' or 'display'")
    try:
        import wcwidth  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Native Unicode rendering requires the 'unicode' extra: "
            "pip install 'bea-tools[unicode]'"
        ) from exc
    return text


def _clip(text: str, width: int, unicode_mode: str) -> str:
    if unicode_mode == "display":
        from wcwidth import wcswidth, wcwidth

        if wcswidth(text) <= width:
            return text
        marker = "." * min(3, width)
        target = max(0, width - len(marker))
        output = []
        used = 0
        for character in text:
            cells = max(0, wcwidth(character))
            if used + cells > target:
                break
            output.append(character)
            used += cells
        return "".join(output) + marker
    if len(text) <= width:
        return text
    if width <= 3:
        return "." * width
    return text[: width - 3] + "..."


def render_plaintext(
    result: ExplorerResult | Mapping[str, Any],
    *,
    width: int = 100,
    max_lines: int = 200,
    max_nodes: int = 1000,
    missing_label: str = "<NA>",
    unicode_mode: str = "safe",
) -> str:
    """Render a bounded, terminal-safe summary without a JSON intermediary."""

    validate_limit("width", width, zero=False)
    validate_limit("max_lines", max_lines, zero=False)
    validate_limit("max_nodes", max_nodes)
    data = result.to_dict() if isinstance(result, ExplorerResult) else result
    lines: list[str] = []

    def add(text: str) -> bool:
        if len(lines) >= max_lines - 1:
            return False
        lines.append(_clip(_safe(text, unicode_mode), width, unicode_mode))
        return True

    add(f"bea-tools feature explorer v{data.get('schema_version', '?')}")
    sections = data.get("sections")
    if sections:
        level_data = sections.get("levels", {})
        census_data = sections.get("census", {})
        grain_data = sections.get("grain", {})
        pair_data = sections.get("pairs", {})
    else:
        kind = data.get("kind")
        level_data = data if kind == "levels" else {}
        census_data = data if kind == "census" else {}
        grain_data = data if kind == "grain" else {}
        pair_data = data if kind == "pairs" else {}
    if level_data:
        add("Levels")
        for feature in level_data.get("per_feature", []):
            column = display_scalar(_identity(feature["column"]), missing_label)
            if not add(
                f"  {column}: {feature['levels_reported']}/{feature['levels_total']} levels"
            ):
                break
            for level in feature.get("levels", []):
                label = display_scalar(_identity(level["value"]), missing_label)
                if not add(f"    {label}: {level['count']}"):
                    break
    if census_data and len(lines) < max_lines - 1:
        tree = census_data.get("tree", {})
        scope = next(iter(census_data.get("scopes", [])), {})
        conditional = ", conditional" if scope.get("conditional") else ""
        add(f"Census ({tree.get('status', census_data.get('status', '?'))}{conditional})")
        dictionary = {
            item["level_id"]: display_scalar(_identity(item["value"]), missing_label)
            for item in census_data.get("level_dictionary", [])
        }
        nodes = tree.get("nodes", [])
        for node in nodes[:max_nodes]:
            indent = "  " * min(node["depth"], max(0, width // 2))
            label = dictionary.get(node["level_id"], node["level_id"])
            suffix = ""
            if node.get("omitted_child_rows"):
                suffix = f" [+{node['omitted_child_rows']} collapsed]"
            if not add(f"{indent}{label}: {node['count']}{suffix}"):
                break
        if len(nodes) > max_nodes:
            add(f"  ... {len(nodes) - max_nodes} nodes not rendered")
    if grain_data and len(lines) < max_lines - 1:
        add("Grain")
        for dependency in grain_data.get("dependencies", []):
            target = display_scalar(_identity(dependency["target"]), missing_label)
            state = dependency["holds"]
            if not add(
                f"  {dependency['key_name']} -> {target}: {state} "
                f"({dependency['violating_groups']} violating groups)"
            ):
                break
    if pair_data and len(lines) < max_lines - 1:
        add("Pairs")
        for pair in pair_data.get("pairs", []):
            if not add(
                f"  {pair['pair'][0]} / {pair['pair'][1]}: "
                f"{pair['relation']}, V={pair['cramers_v']}"
            ):
                break
            absence = pair.get("absence")
            if absence and not add(
                f"    absent: {absence['absent_cells']} "
                f"({len(absence.get('examples', []))} examples)"
            ):
                break
    truncated = len(lines) >= max_lines - 1
    if truncated:
        lines = lines[: max_lines - 1]
        lines.append(_clip("...", width, unicode_mode))
    return "\n".join(lines)
