"""Thin pandas DataFrame accessor delegation."""

from __future__ import annotations

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from .census import census, levels
from .grain import grain
from .orchestration import explore
from .roles import infer_schema


@register_dataframe_accessor("bea")
class BeaDataFrameTools:
    def __init__(self, pandas_object: pd.DataFrame) -> None:
        self._obj = pandas_object

    def infer_schema(self, *args, **kwargs):
        return infer_schema(self._obj, *args, **kwargs)

    def levels(self, *args, **kwargs):
        return levels(self._obj, *args, **kwargs)

    def census(self, *args, **kwargs):
        return census(self._obj, *args, **kwargs)

    def grain(self, *args, **kwargs):
        return grain(self._obj, *args, **kwargs)

    def explore(self, *args, **kwargs):
        return explore(self._obj, *args, **kwargs)
