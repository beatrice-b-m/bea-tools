"""Pandas integrations; optional sampling is lazy."""

from .series import BeaSeriesTools

_SAMPLING_EXPORTS = (
    "LPSampler",
    "FeatureConstraint",
    "HomogeneityConstraint",
    "UniquenessConstraint",
)


def __getattr__(name: str):
    if name in _SAMPLING_EXPORTS:
        from importlib import import_module

        sampler = import_module("bea_tools._pandas.sampler")
        return getattr(sampler, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_SAMPLING_EXPORTS})


__all__ = ["BeaSeriesTools"]
