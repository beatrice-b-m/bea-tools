"""Public package API with optional features loaded only when requested."""

from bea_tools._explore import (
    ExplorerResult,
    KeySpec,
    SchemaProposal,
    census,
    explore,
    grain,
    infer_schema,
    levels,
    render_plaintext,
)

# Accessor modules are deliberately lightweight. Their import makes ordinary
# ``import bea_tools`` sufficient for both Series.bea and DataFrame.bea.
from bea_tools._explore import accessor as _dataframe_accessor  # noqa: F401
from bea_tools._pandas import series as _series_accessor  # noqa: F401
from bea_tools.utility import aligned, divider

_SAMPLING_EXPORTS = (
    "LPSampler",
    "FeatureConstraint",
    "HomogeneityConstraint",
    "UniquenessConstraint",
)
_PLOTTING_EXPORTS = ("bar_plot", "stratified_bar_plot")


def __getattr__(name: str):
    if name in _SAMPLING_EXPORTS:
        try:
            from bea_tools._pandas import sampler
        except ModuleNotFoundError as exc:
            if exc.name == "pulp":
                raise ImportError(
                    "Sampling requires the 'sampling' extra: pip install 'bea-tools[sampling]'"
                ) from exc
            raise
        return getattr(sampler, name)
    if name in _PLOTTING_EXPORTS:
        try:
            from bea_tools._matplotlib import general
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                raise ImportError(
                    "Plotting requires the 'plotting' extra: pip install 'bea-tools[plotting]'"
                ) from exc
            raise
        return getattr(general, name)
    raise AttributeError(f"module 'bea_tools' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SAMPLING_EXPORTS, *_PLOTTING_EXPORTS})


__all__ = [
    "ExplorerResult",
    "KeySpec",
    "SchemaProposal",
    "aligned",
    "census",
    "divider",
    "explore",
    "grain",
    "infer_schema",
    "levels",
    "render_plaintext",
]
